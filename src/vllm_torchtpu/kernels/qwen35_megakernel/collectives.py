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
"""On-chip and cross-chip Pallas remote-DMA all-reduce and residual helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .config import BF16, F32, rnd
from .context import MKBuildContext


class PallasAllReduce:
    """Remote-DMA all-reduce supporting HIER=0, 1, 2, 3 and RSAG."""

    def __init__(self, ctx: MKBuildContext):
        self.ctx = ctx

    def ar_copies(self, slot, rnd_i: int = 0, half=None):
        """Remote copies of all-reduce round rnd_i of this slot to the peers me ^ off. A peer's
        copy to this rank uses the same off, so it lands on the same semaphore and size."""
        ctx = self.ctx
        HIER, RSAG, R = ctx.HIER, ctx.RSAG, ctx.R
        HC2, ARP_H1, AR0S = ctx.HC2, ctx.ARP_H1, ctx.AR0S
        scr, me = ctx.scr, ctx.me
        if HIER == 3:
            off = 1 << rnd_i
            bit = lax.rem(me // off, 2)
            return [
                pltpu.make_async_remote_copy(
                    src_ref=scr["send"].at[rnd_i, slot],
                    dst_ref=scr["recv"].at[rnd_i, slot, bit],
                    send_sem=scr["ar_ssem"].at[rnd_i, slot, 0],
                    recv_sem=scr["ar_rsem"].at[rnd_i, slot, 0],
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            ]
        if HIER in (1, 2):
            c_idx = lax.rem(me, 2)
            if rnd_i == 0:
                src = scr["send"].at[slot, 1 - c_idx] if HIER == 2 else scr["send"].at[slot]
                peers = [(0, 1)]
                dst = scr["recv"].at[slot, c_idx]
                if HIER == 1 and AR0S and slot == 0:
                    hs_list = (0, 1) if half is None else (half,)
                    return [
                        pltpu.make_async_remote_copy(
                            src_ref=scr["send"].at[slot, :, pl.ds(i * HC2, HC2)],
                            dst_ref=scr["recv"].at[slot, c_idx, :, pl.ds(i * HC2, HC2)],
                            send_sem=scr["ar_ssem"].at[0, slot, i],
                            recv_sem=scr["ar_rsem"].at[0, slot, i],
                            device_id=(me ^ 1,),
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                        for i in hs_list
                    ]
            elif rnd_i == 1 or (HIER == 1 and rnd_i == 2):
                src = scr["ag_send"].at[slot]
                cross_offs = (3, 5, 7) if (HIER == 1 and R == 8) else (2, 4, 6)
                peers = list(enumerate(cross_offs))
                dst = scr["ag_recv"].at[slot, me // 2]
                if HIER == 1 and ARP_H1 and (slot == 1 or AR0S):
                    hl = pl.ds((rnd_i - 1) * HC2, HC2)
                    src = scr["ag_send"].at[slot, :, hl]
                    dst = scr["ag_recv"].at[slot, me // 2, :, hl]
            else:
                src = scr["s2_send"].at[slot]
                peers = [(0, 1)]
                dst = scr["s2_recv"].at[slot, c_idx]
            return [
                pltpu.make_async_remote_copy(
                    src_ref=src,
                    dst_ref=dst,
                    send_sem=scr["ar_ssem"].at[rnd_i, slot, i],
                    recv_sem=scr["ar_rsem"].at[rnd_i, slot, i],
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                for i, off in peers
            ]
        if RSAG and rnd_i == 0:  # lane chunk p of this rank's partial -> rank p
            src, dst = (lambda off: scr["send"].at[slot, me ^ off]), scr["recv"].at[slot, me]
        elif RSAG:  # this rank's chunk sum -> every peer
            src, dst = (lambda off: scr["ag_send"].at[slot]), scr["ag_recv"].at[slot, me]
        else:  # the whole partial -> every peer
            src, dst = (lambda off: scr["send"].at[slot]), scr["recv"].at[slot, me]
        return [
            pltpu.make_async_remote_copy(
                src_ref=src(off),
                dst_ref=dst,
                send_sem=scr["ar_ssem"].at[rnd_i, slot, off - 1],
                recv_sem=scr["ar_rsem"].at[rnd_i, slot, off - 1],
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for off in range(1, R)
        ]

    def ar_rounds(self, slot):
        ctx = self.ctx
        return (
            (0, 1)
            if (ctx.ARP_H1 and slot == 0 and not ctx.AR0S)
            else ctx.AR_ROUNDS
        )

    def ar_round(self, slot, rnd_i: int, fill=None):
        descs = self.ar_copies(slot, rnd_i)
        for d in descs:
            d.start()
        if fill is not None:
            fill()
        for d in descs:
            d.wait_recv()

    def allreduce(
        self,
        slot,
        x,
        first,
        fill=None,
        defer: bool = False,
        pre_sent: bool = False,
    ):
        """x [M, H] f32 partial -> the sum over ranks of its B live rows ([MR, H] when WAR,
        else folded [PR, PC]). Only the incoming payloads are waited for; this slot's sends are
        waited for when its send buffers are next rewritten (not first) and at kernel exit."""
        ctx = self.ctx
        WAR, MR, B, PC, HPC, PR = ctx.WAR, ctx.MR, ctx.B, ctx.PC, ctx.HPC, ctx.PR
        HIER, N_HIER3, RSAG, R, CW, HC2 = (
            ctx.HIER,
            ctx.N_HIER3,
            ctx.RSAG,
            ctx.R,
            ctx.CW,
            ctx.HC2,
        )
        ARP_H1, AR0S, comm = ctx.ARP_H1, ctx.AR0S, ctx.comm
        scr, me, sc = ctx.scr, ctx.me, ctx.sc
        send, recv = scr["send"], scr["recv"]
        c_idx = lax.rem(me, 2)
        if pre_sent:
            assert HIER == 2 and comm and fill is None and not defer
            with sc("ar_comm"):
                for d in self.ar_copies(slot, 0):
                    d.wait_recv()
        else:
            with sc("ar_pack"):
                if WAR:
                    val = x[:MR]
                else:
                    chunks = [x[0:B, j * PC : (j + 1) * PC] for j in range(HPC)]
                    if B * HPC < PR:
                        chunks.append(jnp.zeros((PR - B * HPC, PC), F32))
                    val = jnp.concatenate(chunks, axis=0)
                if comm:
                    @pl.when(jnp.logical_not(first))
                    def _():
                        for rnd_i in self.ar_rounds(slot):
                            for d in self.ar_copies(slot, rnd_i):
                                d.wait_send()

                    if defer:
                        @pl.when(first)
                        def _():
                            for d in self.ar_copies(slot, 2):
                                d.wait_send()

                if HIER == 3:
                    send[0, slot] = val.astype(BF16)
                    recv[0, slot, c_idx] = send[0, slot]
                elif HIER == 2:
                    send[slot, 0] = val[:, :HC2].astype(BF16)
                    send[slot, 1] = val[:, HC2:].astype(BF16)
                    recv[slot, c_idx] = send[slot, c_idx]
                elif HIER == 1:
                    send[slot] = val.astype(BF16)
                    recv[slot, c_idx] = send[slot]
                elif RSAG:
                    for j in range(R):
                        send[slot, j] = val[:, j * CW : (j + 1) * CW].astype(BF16)
                    recv[slot, me] = send[slot, me]
                else:
                    send[slot] = val.astype(BF16)
                    recv[slot, me] = send[slot]
            if not comm:
                if fill is not None:
                    fill()
                return rnd(val)
            with sc("ar_comm"):
                self.ar_round(slot, 0, fill if HIER == 0 and not RSAG else None)
        if HIER == 3:
            for rnd_i in range(1, N_HIER3):
                with sc("ar_sum"):
                    b_i = lax.rem(me // (1 << rnd_i), 2)
                    send[rnd_i, slot] = (
                        recv[rnd_i - 1, slot, 0].astype(F32)
                        + recv[rnd_i - 1, slot, 1].astype(F32)
                    ).astype(BF16)
                    recv[rnd_i, slot, b_i] = send[rnd_i, slot]
                with sc("ar_comm"):
                    self.ar_round(slot, rnd_i, fill=fill if rnd_i == 1 else None)
            with sc("ar_sum"):
                return (
                    recv[N_HIER3 - 1, slot, 0].astype(F32)
                    + recv[N_HIER3 - 1, slot, 1].astype(F32)
                )
        if HIER in (1, 2):
            ag_send, ag_recv = scr["ag_send"], scr["ag_recv"]
            with sc("ar_sum"):
                ag_send[slot] = (
                    recv[slot, 0].astype(F32) + recv[slot, 1].astype(F32)
                ).astype(BF16)
                ag_recv[slot, me // 2] = ag_send[slot]
            if HIER == 1 and defer:
                assert ctx.ARP and slot == 1
                with sc("ar_comm"):
                    h0 = self.ar_copies(slot, 1)
                    for d in h0 + self.ar_copies(slot, 2):
                        d.start()
                    if fill is not None:
                        fill()
                    for d in h0:
                        d.wait_recv()
                with sc("ar_res"):
                    self.arp_half_add(
                        jnp.sum(ag_recv[slot, :, :, 0:HC2].astype(F32), axis=0), 0
                    )
                return None
            with sc("ar_comm"):
                self.ar_round(slot, 1, fill=fill)
            if HIER == 1:
                with sc("ar_sum"):
                    return jnp.sum(ag_recv[slot].astype(F32), axis=0)
            s2_send, s2_recv = scr["s2_send"], scr["s2_recv"]
            with sc("ar_sum"):
                s2_send[slot] = jnp.sum(ag_recv[slot].astype(F32), axis=0).astype(BF16)
                if not defer:
                    s2_recv[slot, c_idx] = s2_send[slot]
            if defer:
                assert HIER == 2 and ctx.ARP
                with sc("ar_comm"):
                    for d in self.ar_copies(slot, 2):
                        d.start()
                with sc("ar_res"):
                    self.arp_half_add(s2_send[slot].astype(F32), ctx.arp_c)
                return None
            with sc("ar_comm"):
                self.ar_round(slot, 2)
            with sc("ar_sum"):
                return jnp.concatenate(
                    [s2_recv[slot, 0], s2_recv[slot, 1]], axis=-1
                ).astype(F32)
        if not RSAG:
            with sc("ar_sum"):
                return jnp.sum(recv[slot].astype(F32), axis=0)
        ag_send, ag_recv = scr["ag_send"], scr["ag_recv"]
        with sc("ar_sum"):
            ag_send[slot] = jnp.sum(recv[slot].astype(F32), axis=0).astype(BF16)
            ag_recv[slot, me] = ag_send[slot]
        with sc("ar_comm"):
            self.ar_round(slot, 1, fill=fill)
        with sc("ar_sum"):
            return jnp.concatenate([ag_recv[slot, j] for j in range(R)], axis=-1).astype(F32)

    def add_residual(self, tot):
        ctx = self.ctx
        B, HPC, WAR, scr, sc = ctx.B, ctx.HPC, ctx.WAR, ctx.scr, ctx.sc
        res = scr["res"]
        with sc("ar_res"):
            if not WAR:
                tot = jnp.concatenate([tot[j * B : (j + 1) * B] for j in range(HPC)], axis=-1)
            res[0:B, :] = rnd(res[0:B, :] + tot[0:B])

    def arp_lanes(self, j, h):
        """First lane of payload half h (traced: the core index) in fold block j."""
        ctx = self.ctx
        AR_C, HC2 = ctx.AR_C, ctx.HC2
        return pl.multiple_of(j * AR_C + h * HC2, HC2)

    def arp_half_add(self, val, h):
        """add_residual for payload half h only (val: f32 [AR_R, HC2] payload layout). h is
        traced (runtime lane offset, no control flow)."""
        ctx = self.ctx
        B, H, AR_C, HC2, WAR = ctx.B, ctx.H, ctx.AR_C, ctx.HC2, ctx.WAR
        res = ctx.scr["res"]
        with ctx.sc("ar_res"):
            for j in range(H // AR_C):
                l0 = self.arp_lanes(j, h)
                r0 = 0 if WAR else j * B
                res[0:B, pl.ds(l0, HC2)] = rnd(
                    res[0:B, pl.ds(l0, HC2)] + val[r0 : r0 + B]
                )

    def arp_norm_half(self, slot, h, first: bool):
        """Input RMSNorm on payload half h's lanes: xb = x * (1 + w) unscaled (r is applied
        after the matmuls); first: arp = sum(x^2) of the half, else arp = r over both."""
        ctx = self.ctx
        c, H, AR_C, HC2, MB = ctx.c, ctx.H, ctx.AR_C, ctx.HC2, ctx.MB
        scr, lay = ctx.scr, ctx.lay
        o_w = lay["in_norm"][0]
        ss = None
        for j in range(H // AR_C):
            l0 = self.arp_lanes(j, h)
            x = scr["res"][:, pl.ds(l0, HC2)]
            w_ = scr["sm"][slot, :, pl.ds(o_w + l0, HC2)]
            scr["xb"][:, pl.ds(l0, HC2)] = (x * (1.0 + w_)).astype(BF16)
            s_ = jnp.sum(x * x, axis=-1, keepdims=True)
            ss = s_ if ss is None else ss + s_
        if first:
            scr["arp"][...] = jnp.broadcast_to(ss, (MB, 128))
        else:
            ms = (scr["arp"][:, 0:1] + ss) * (1.0 / H)
            scr["arp"][...] = jnp.broadcast_to(lax.rsqrt(ms + c.eps), (MB, 128))

    def arp_round2_add(self):
        """Wait the deferred MoE all-reduce's round 2; add the partner half's residual
        (hier=1: push half 1, summed over the 4 chips like the unsplit all-reduce)."""
        ctx = self.ctx
        HIER, HC2, scr = ctx.HIER, ctx.HC2, ctx.scr
        for d in self.ar_copies(1, 2):
            d.wait_recv()
        if HIER == 1:
            self.arp_half_add(
                jnp.sum(scr["ag_recv"][1, :, :, HC2 : 2 * HC2].astype(F32), axis=0), 1
            )
        else:
            self.arp_half_add(
                scr["s2_recv"][1, 1 - ctx.arp_c].astype(F32), 1 - ctx.arp_c
            )

    def ar0_cols(self, h):
        """AR0S: (c0, A0_CW) out-projection column chunks of payload half h (int or traced)."""
        ctx = self.ctx
        H, AR_C, HC2, A0_CW = ctx.H, ctx.AR_C, ctx.HC2, ctx.A0_CW
        cols = []
        for j in range(H // AR_C):
            for i in range(HC2 // A0_CW):
                c0 = j * AR_C + h * HC2 + i * A0_CW
                if not isinstance(c0, int):
                    c0 = pl.multiple_of(c0, A0_CW)
                cols.append((c0, A0_CW))
        return cols

    def ar0_payload(self, lslot, h, vals):
        """AR0S: f32 [AR_R, HC2] payload half h of the out projection * out_s (= lanes
        h * HC2 .. of the allreduce val); vals: the half's chunk sums in ar0_cols(h) order."""
        ctx = self.ctx
        WAR, MR, B, PR, HPC, H, AR_C, HC2, A0_CW = (
            ctx.WAR,
            ctx.MR,
            ctx.B,
            ctx.PR,
            ctx.HPC,
            ctx.H,
            ctx.AR_C,
            ctx.HC2,
            ctx.A0_CW,
        )
        scr, lay = ctx.scr, ctx.lay
        o_s = lay["out_s"][0]
        rows = MR if WAR else B
        nc = HC2 // A0_CW
        blocks = []
        for j in range(H // AR_C):
            l0 = j * AR_C + h * HC2
            if not isinstance(l0, int):
                l0 = pl.multiple_of(l0, HC2)
            yb = [v[0:rows] for v in vals[j * nc : (j + 1) * nc]]
            yb = yb[0] if nc == 1 else jnp.concatenate(yb, axis=-1)
            blocks.append(yb * scr["sm"][lslot, :, pl.ds(o_s + l0, HC2)])
        if WAR:
            return blocks[0]
        if B * HPC < PR:
            blocks.append(jnp.zeros((PR - B * HPC, HC2), F32))
        return jnp.concatenate(blocks, axis=0)

    def ar0_spec(self, lslot, first):
        """AR0S producer spec for dmatmul(split=...): pre() before the out projection's
        chunks, then half a chunks + hook_a, then half b chunks + hook_b."""
        ctx = self.ctx
        HIER, HC2 = ctx.HIER, ctx.HC2
        scr, me, sc = ctx.scr, ctx.me, ctx.sc
        slot = 0
        send, recv = scr["send"], scr["recv"]
        c_idx = lax.rem(me, 2)

        def pre():
            @pl.when(jnp.logical_not(first))
            def _():
                for rnd_i in self.ar_rounds(slot):
                    for d in self.ar_copies(slot, rnd_i):
                        d.wait_send()

        if HIER == 2:
            h_a, h_b = 1 - c_idx, c_idx

            def hook_a(vals):
                with sc("ar_pack"):
                    send[slot, 1 - c_idx] = self.ar0_payload(
                        lslot, 1 - c_idx, vals
                    ).astype(BF16)
                with sc("ar_comm"):
                    for d in self.ar_copies(slot, 0):
                        d.start()

            def hook_b(vals):
                with sc("ar_pack"):
                    recv[slot, c_idx] = self.ar0_payload(lslot, c_idx, vals).astype(
                        BF16
                    )
        else:
            h_a, h_b = 0, 1

            def hook_h(h, vals):
                hl = slice(h * HC2, (h + 1) * HC2)
                with sc("ar_pack"):
                    v = self.ar0_payload(lslot, h, vals).astype(BF16)
                    send[slot, :, hl] = v
                    recv[slot, c_idx, :, hl] = v
                with sc("ar_comm"):
                    self.ar_copies(slot, 0, half=h)[0].start()

            hook_a, hook_b = (lambda v: hook_h(0, v)), (lambda v: hook_h(1, v))
        return {
            "pre": pre,
            "cols": (self.ar0_cols(h_a), self.ar0_cols(h_b)),
            "hooks": (hook_a, hook_b),
        }

    def ar0_finish(self, slot, first):
        """AR0S: rest of the mixer all-reduce after the producer's sends, residual added."""
        ctx = self.ctx
        HIER, HC2 = ctx.HIER, ctx.HC2
        scr, me, sc = ctx.scr, ctx.me, ctx.sc
        if HIER == 2:
            self.add_residual(self.allreduce(slot, None, first, pre_sent=True))
            return
        recv, ag_send, ag_recv = scr["recv"], scr["ag_send"], scr["ag_recv"]
        for h in range(2):
            hl = slice(h * HC2, (h + 1) * HC2)
            with sc("ar_comm"):
                self.ar_copies(slot, 0, half=h)[0].wait_recv()
            with sc("ar_sum"):
                v = (
                    recv[slot, 0, :, hl].astype(F32)
                    + recv[slot, 1, :, hl].astype(F32)
                ).astype(BF16)
                ag_send[slot, :, hl] = v
                ag_recv[slot, me // 2, :, hl] = v
            with sc("ar_comm"):
                for d in self.ar_copies(slot, 1 + h):
                    d.start()
        for h in range(2):
            with sc("ar_comm"):
                for d in self.ar_copies(slot, 1 + h):
                    d.wait_recv()
            self.arp_half_add(
                jnp.sum(
                    ag_recv[slot, :, :, h * HC2 : (h + 1) * HC2].astype(F32), axis=0
                ),
                h,
            )

    def wait_sends(self):
        """Wait for all in-flight all-reduce sends at kernel exit."""
        ctx = self.ctx
        if ctx.comm:
            for ar_slot in range(2):
                for rnd_i in self.ar_rounds(ar_slot):
                    for d in self.ar_copies(ar_slot, rnd_i):
                        d.wait_send()
