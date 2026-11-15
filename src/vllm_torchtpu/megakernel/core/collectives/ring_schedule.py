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
"""`AllReduceRing` helpers for the schedule options: the `hx` round copies,
the group exchange, the deferred all-reduce round (`arpipe`) and round-0 sends
from a column split (`ar0s`). Split out of `ring.py` (package size cap)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..ops import BF16, F32, round_bf16
from .hooks import ColumnSplit


class AllReduceScheduleMixin:
    """Methods of `AllReduceRing` (`cp`, `scr`, `me`, `sc`, `comm` are its)."""

    def _hx_copies(self, slot, rnd_i: int):
        """`hx` remote copies of rounds 1 (4-chip RS) and 3 (4-chip AG)."""
        cp, scr, me = self.cp, self.scr, self.me
        # hx: round 1 sends quarter q of this core's half-sum to the core of
        # chip q (4-chip RS); round 3 sends this chip's summed quarter to the
        # other three (4-chip AG), into the same lanes of their s2_send.
        q_w = cp.HC2 // 4
        chip = me // 2
        out = []
        for i, off in enumerate((2, 4, 6)):
            peer = me ^ off
            if rnd_i == 1:
                lo = pl.multiple_of((peer // 2) * q_w, q_w)
                src = scr["ag_send"].at[slot, :, pl.ds(lo, q_w)]
                dst = scr["ag_recv"].at[slot, chip, :, pl.ds(0, q_w)]
            else:
                lo = pl.multiple_of(chip * q_w, q_w)
                src = scr["s2_send"].at[slot, :, pl.ds(lo, q_w)]
                dst = scr["s2_send"].at[slot, :, pl.ds(lo, q_w)]
            out.append(
                pltpu.make_async_remote_copy(
                    src_ref=src,
                    dst_ref=dst,
                    send_sem=scr["ar_ssem"].at[rnd_i, slot, i],
                    recv_sem=scr["ar_rsem"].at[rnd_i, slot, i],
                    device_id=(peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            )
        return out

    def group_copies(self, src, dst, ssem, rsem, G: int):
        """Remote copies `src(off) -> dst` on group peer `me ^ off` (`1..G-1`)."""
        me = self.me
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

    @staticmethod
    def exchange(descs):
        """Start `descs` and wait on both `wait_recv()` and `wait_send()`."""
        for d in descs:
            d.start()
        for d in descs:
            d.wait_recv()
        for d in descs:
            d.wait_send()

    # ------------------------------------------------------------------
    # ARPIPE: the deferred all-reduce (`cp.ARP`)
    # ------------------------------------------------------------------
    def arp_lanes(self, j: int, h: Any) -> Any:
        """First lane of payload half `h` (may be traced) in fold block `j`."""
        cp = self.cp
        return pl.multiple_of(j * cp.AR_C + h * cp.HC2, cp.HC2)

    def arp_half_add(self, val: Any, h: Any) -> None:
        """`add_residual` for payload half `h` only (`val`: f32 `[AR_R, HC2]` in
        payload layout). `h` may be traced (a run-time lane offset)."""
        cp, scr = self.cp, self.scr
        B, HC2 = cp.B, cp.HC2
        res = scr["res"]
        with self.sc("ar_res"):
            for j in range(cp.H // cp.AR_C):
                l0 = self.arp_lanes(j, h)
                r0 = 0 if cp.WAR else j * B
                res[0:B, pl.ds(l0, HC2)] = round_bf16(
                    res[0:B, pl.ds(l0, HC2)] + val[r0 : r0 + B]
                )

    def arp_norm_half(
        self,
        w_off: int,
        slot: Any,
        h: Any,
        first: bool,
        *,
        eps: float,
        zero_centered: bool,
    ) -> None:
        """Input RMSNorm of payload half `h`'s lanes, left unscaled.

        `xb = x * w` (`1 + w` when `zero_centered`; `w` = the norm weights at
        lane offset `w_off` of `scr['sm'][slot]`). `first`: `scr['arp']` = the
        half's sum of squares; else `scr['arp']` = the reciprocal RMS over both
        halves, applied after the input projection (`KernelContext.arp_scale`).
        """
        cp, scr = self.cp, self.scr
        H, HC2 = cp.H, cp.HC2
        arp = scr["arp"]
        ss = None
        for j in range(H // cp.AR_C):
            l0 = self.arp_lanes(j, h)
            x = scr["res"][:, pl.ds(l0, HC2)]
            w_ = scr["sm"][slot, :, pl.ds(w_off + l0, HC2)]
            scale = (1.0 + w_) if zero_centered else w_
            scr["xb"][:, pl.ds(l0, HC2)] = (x * scale).astype(BF16)
            s_ = jnp.sum(x * x, axis=-1, keepdims=True)
            ss = s_ if ss is None else ss + s_
        if first:
            arp[...] = jnp.broadcast_to(ss, arp.shape)
        else:
            ms = (arp[:, 0:1] + ss) * (1.0 / H)
            arp[...] = jnp.broadcast_to(lax.rsqrt(ms + eps), arp.shape)

    def arp_round2_add(self) -> None:
        """Wait for the deferred all-reduce's last round and add the other
        payload half to the residual (hier=1: half 1, summed over the 4 chips
        in chip order like the undeferred all-reduce)."""
        cp, scr = self.cp, self.scr
        HC2 = cp.HC2
        for d in self.ar_copies(1, 2):
            d.wait_recv()
        if cp.HIER == 1:
            self.arp_half_add(
                jnp.sum(scr["ag_recv"][1, :, :, HC2 : 2 * HC2].astype(F32), axis=0), 1
            )
        else:
            self.arp_half_add(
                scr["s2_recv"][1, 1 - self.arp_c].astype(F32), 1 - self.arp_c
            )

    def arp_prime(self) -> None:
        """Send a zero last round on buffer slot 1 before the first layer, so
        layer 0's `arp_round2_add` has a (zero) deferred half to add."""
        cp, scr, me = self.cp, self.scr, self.me
        self._used_slots.add(1)
        if cp.HIER == 1:  # zero push of half 1 from every chip, own entry included
            scr["ag_send"][1] = jnp.zeros(scr["ag_send"].shape[1:], BF16)
            scr["ag_recv"][1, me // 2] = jnp.zeros(scr["ag_recv"].shape[2:], BF16)
        else:
            scr["s2_send"][1] = jnp.zeros(scr["s2_send"].shape[1:], BF16)
        for d in self.ar_copies(1, 2):
            d.start()

    # ------------------------------------------------------------------
    # AR0S: round 0 sent from inside the producer's output projection
    # ------------------------------------------------------------------
    def ar0_cols(self, h: Any) -> list[tuple[Any, int]]:
        """`(c0, A0_CW)` output-projection column chunks of payload half `h`
        (an int or traced), in payload order."""
        cp = self.cp
        AR_C, HC2, A0_CW = cp.AR_C, cp.HC2, cp.A0_CW
        cols = []
        for j in range(cp.H // AR_C):
            for i in range(HC2 // A0_CW):
                c0 = j * AR_C + h * HC2 + i * A0_CW
                if not isinstance(c0, int):
                    c0 = pl.multiple_of(c0, A0_CW)
                cols.append((c0, A0_CW))
        return cols

    def ar0_payload(
        self, lslot: Any, h: Any, vals: list[Any], scale_off: int | None
    ) -> Any:
        """f32 `[AR_R, HC2]` payload half `h` of the all-reduce input from the
        output projection's chunk sums `vals` (in `ar0_cols(h)` order), times
        the output scale at lane offset `scale_off` of `scr['sm'][lslot]` (no
        scale when `None`)."""
        cp, scr = self.cp, self.scr
        WAR, B, PR, HPC, AR_C, HC2 = cp.WAR, cp.B, cp.PR, cp.HPC, cp.AR_C, cp.HC2
        rows = cp.MR if WAR else B
        nc = HC2 // cp.A0_CW
        blocks = []
        for j in range(cp.H // AR_C):
            l0 = j * AR_C + h * HC2
            if not isinstance(l0, int):
                l0 = pl.multiple_of(l0, HC2)
            yb = [v[0:rows] for v in vals[j * nc : (j + 1) * nc]]
            yb = yb[0] if nc == 1 else jnp.concatenate(yb, axis=-1)
            if scale_off is not None:
                yb = yb * scr["sm"][lslot, :, pl.ds(scale_off + l0, HC2)]
            blocks.append(yb)
        if WAR:
            return blocks[0]
        if B * HPC < PR:
            blocks.append(jnp.zeros((PR - B * HPC, HC2), F32))
        return jnp.concatenate(blocks, axis=0)

    def ar0_spec(
        self, lslot: Any, first: Any, slot: int = 0, scale_off: int | None = None
    ) -> ColumnSplit:
        """The `split=` for the producer's output projection (`TwoRing.dmatmul`).

        `pre()` waits for buffer `slot`'s previous sends; the first column
        group is the payload half sent first, and its hook packs and sends it
        (hier=2: the half for the on-chip twin; hier=1: half 0); the second
        group's hook packs the other half (hier=1: and sends it). `lslot`: the
        smalls slot of the output scale at lane offset `scale_off`.
        """

        cp, scr, me, sc = self.cp, self.scr, self.me, self.sc
        HC2 = cp.HC2
        send, recv = scr["send"], scr["recv"]
        c_idx = lax.rem(me, 2)
        self._used_slots.add(slot)

        def pre() -> None:
            @pl.when(jnp.logical_not(first))
            def _():
                for rnd_i in self.ar_rounds(slot):
                    for d in self.ar_copies(slot, rnd_i):
                        d.wait_send()

        if cp.HIER == 2:
            h_a, h_b = 1 - c_idx, c_idx

            def hook_a(vals: list[Any]) -> None:
                with sc("ar_pack"):
                    send[slot, 1 - c_idx] = self.ar0_payload(
                        lslot, 1 - c_idx, vals, scale_off
                    ).astype(BF16)
                with sc("ar_comm"):
                    for d in self.ar_copies(slot, 0):
                        d.start()

            def hook_b(vals: list[Any]) -> None:
                with sc("ar_pack"):
                    recv[slot, c_idx] = self.ar0_payload(
                        lslot, c_idx, vals, scale_off
                    ).astype(BF16)

        else:
            h_a, h_b = 0, 1

            def hook_h(h: int, vals: list[Any]) -> None:
                hl = slice(h * HC2, (h + 1) * HC2)
                with sc("ar_pack"):
                    v = self.ar0_payload(lslot, h, vals, scale_off).astype(BF16)
                    send[slot, :, hl] = v
                    recv[slot, c_idx, :, hl] = v
                with sc("ar_comm"):
                    self.ar_copies(slot, 0, half=h)[0].start()

            def hook_a(vals: list[Any]) -> None:
                hook_h(0, vals)

            def hook_b(vals: list[Any]) -> None:
                hook_h(1, vals)

        return ColumnSplit(
            pre, (self.ar0_cols(h_a), self.ar0_cols(h_b)), (hook_a, hook_b)
        )

    def ar0_finish(self, slot: int, first: Any) -> None:
        """The rest of the all-reduce after `ar0_spec`'s sends, residual added."""
        cp, scr, me, sc = self.cp, self.scr, self.me, self.sc
        HC2 = cp.HC2
        if cp.HIER == 2:
            self.add_residual(self.allreduce(slot, None, first, pre_sent=True))
            return
        recv, ag_send, ag_recv = scr["recv"], scr["ag_send"], scr["ag_recv"]
        for h in range(2):
            hl = slice(h * HC2, (h + 1) * HC2)
            with sc("ar_comm"):
                self.ar_copies(slot, 0, half=h)[0].wait_recv()
            with sc("ar_sum"):
                v = (
                    recv[slot, 0, :, hl].astype(F32) + recv[slot, 1, :, hl].astype(F32)
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
