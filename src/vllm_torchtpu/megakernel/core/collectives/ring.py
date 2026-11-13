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
"""ICI remote-DMA collectives (`AllReduceRing`: flat, RS+AG, hierarchical)."""

from __future__ import annotations

import contextlib
from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..ops import BF16, F32, round_bf16
from .params import CommParams
from .ring_schedule import AllReduceScheduleMixin


class AllReduceRing(AllReduceScheduleMixin):
    """In-kernel ICI remote-DMA all-reduce and group exchange primitives.

    The `hx` round copies, the group exchange and the deferred-round (`arpipe`)
    and round-0 (`ar0s`) helpers live in `ring_schedule.AllReduceScheduleMixin`.
    """

    def __init__(
        self,
        cp: CommParams,
        scr: dict[str, Any],
        me: Any,
        *,
        skip: frozenset[str] = frozenset(),
        sc: Any = None,
    ):
        self.cp = cp
        self.scr = scr
        self.me = me
        self.skip = skip
        self.sc = sc if sc is not None else (lambda *_: contextlib.nullcontext())
        self.comm = cp.R > 1 and "comm" not in skip
        self._used_slots: set[int] = set()
        # `cp.ARP`: payload half this core owns first after the deferred
        # all-reduce (hier=2: its core index; hier=1: half 0 on every core).
        self.arp_c: Any = None
        if cp.ARP:
            self.arp_c = jnp.int32(0) if cp.HIER == 1 else lax.rem(me, 2)

    def ar_copies(self, slot, rnd_i: int, half: int | None = None):
        """Remote copy descriptors for round `rnd_i` on double-buffer `slot`.

        `half` (AR0S, hier=1, round 0 of slot 0): only that payload half's copy.
        """
        cp, scr, me = self.cp, self.scr, self.me
        HIER, RSAG, R = cp.HIER, cp.RSAG, cp.R
        if HIER == 3:
            off = 1 << rnd_i
            bit = lax.rem(me >> rnd_i, 2)
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
        if HIER == 2 and cp.HX and rnd_i in (1, 3):
            return self._hx_copies(slot, rnd_i)
        if HIER in (1, 2):
            c_idx = lax.rem(me, 2)
            if rnd_i == 0:
                src = (
                    scr["send"].at[slot, 1 - c_idx]
                    if HIER == 2
                    else scr["send"].at[slot]
                )
                peers = [(0, 1)]
                dst = scr["recv"].at[slot, c_idx]
                if HIER == 1 and cp.AR0S and slot == 0:
                    # AR0S: one on-chip copy per payload half, each sent as
                    # soon as the producer has computed that half
                    HC2 = cp.HC2
                    return [
                        pltpu.make_async_remote_copy(
                            src_ref=scr["send"].at[slot, :, pl.ds(i * HC2, HC2)],
                            dst_ref=scr["recv"].at[slot, c_idx, :, pl.ds(i * HC2, HC2)],
                            send_sem=scr["ar_ssem"].at[0, slot, i],
                            recv_sem=scr["ar_rsem"].at[0, slot, i],
                            device_id=(me ^ 1,),
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                        for i in ((0, 1) if half is None else (half,))
                    ]
            elif rnd_i == 1 or (HIER == 1 and rnd_i == 2):
                src = scr["ag_send"].at[slot]
                peers = list(enumerate(cp.HIER1_PEERS if HIER == 1 else (2, 4, 6)))
                dst = scr["ag_recv"].at[slot, me // 2]
                if HIER == 1 and cp.ARP_H1 and (slot == 1 or cp.AR0S):
                    # split hier=1 all-reduce (deferred slot 1; slot 0 under
                    # AR0S): round 1 pushes payload half 0, round 2 half 1
                    hl = pl.ds((rnd_i - 1) * cp.HC2, cp.HC2)
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
        if RSAG and rnd_i == 0:
            src, dst = (
                (lambda off: scr["send"].at[slot, me ^ off]),
                scr["recv"].at[slot, me],
            )
        elif RSAG:
            src, dst = (
                (lambda off: scr["ag_send"].at[slot]),
                scr["ag_recv"].at[slot, me],
            )
        else:
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

    def ar_rounds(self, slot) -> tuple[int, ...]:
        """Rounds whose sends a use of buffer `slot` issues (`AR_ROUNDS`, except
        the undeferred slot 0 of a split hier=1 schedule: rounds 0 and 1)."""
        cp = self.cp
        if cp.ARP_H1 and slot == 0 and not cp.AR0S:
            return (0, 1)
        return cp.AR_ROUNDS

    def ar_round(self, slot, rnd_i: int, fill=None):
        """Start remote copies for round `rnd_i` on `slot` and wait on `wait_recv()`.

        `fill()` (optional) runs between the starts and the waits.
        """
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
        """Reduce-sum `x [MB, H]` f32 partial across all `R` ranks on `slot`.

        `fill()` (optional, e.g. a deferred dense-ring refill) is issued once
        inside the all-reduce, while its first cross-chip round is in flight
        (round 0 of the flat one-shot, round 1 otherwise), or before returning
        when there is no communication.

        `defer` (`cp.ARP`, buffer slot 1): add only this core's payload half
        (`arp_c`) to the residual and return `None`, leaving the last round in
        flight; `arp_round2_add` (inside the next layer's input projection)
        waits for it and adds the other half.

        `pre_sent` (AR0S, hier=2; `ar0_finish`): round 0 was already packed
        and sent by the producer (`ar0_spec`), so `x` is unused.
        """
        self._used_slots.update((slot,) if isinstance(slot, int) else (0, 1))
        cp, scr, me, sc = self.cp, self.scr, self.me, self.sc
        WAR, MR, B, HPC, PR, PC = cp.WAR, cp.MR, cp.B, cp.HPC, cp.PR, cp.PC
        HIER, RSAG, R, CW, HC2 = cp.HIER, cp.RSAG, cp.R, cp.CW, cp.HC2
        send, recv = scr["send"], scr["recv"]
        if pre_sent:
            assert HIER == 2 and self.comm and fill is None and not defer
            with sc("ar_comm"):
                for d in self.ar_copies(slot, 0):
                    d.wait_recv()
            c_idx = lax.rem(me, 2)
            return self._allreduce_rest(slot, first, fill, defer, c_idx)
        with sc("ar_pack"):
            if WAR:
                val = x[:MR]
            else:
                chunks = [x[0:B, j * PC : (j + 1) * PC] for j in range(HPC)]
                if B * HPC < PR:
                    chunks.append(jnp.zeros((PR - B * HPC, PC), F32))
                val = jnp.concatenate(chunks, axis=0)
            if self.comm:

                @pl.when(jnp.logical_not(first))
                def _():
                    for rnd_i in self.ar_rounds(slot):
                        for d in self.ar_copies(slot, rnd_i):
                            d.wait_send()

                if defer:  # the primed zero round 2 (`arp_prime`)

                    @pl.when(first)
                    def _():
                        for d in self.ar_copies(slot, 2):
                            d.wait_send()

            c_idx = lax.rem(me, 2)
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
        if not self.comm:
            if fill is not None:
                fill()
            return round_bf16(val)
        with sc("ar_comm"):
            # one-round modes take `fill` here; the others in their round 1
            one_round = (HIER == 0 and not RSAG) or (HIER == 3 and cp.N_HIER3 == 1)
            self.ar_round(slot, 0, fill if one_round else None)
        return self._allreduce_rest(slot, first, fill, defer, c_idx)

    def _allreduce_rest(self, slot, first, fill, defer: bool, c_idx):
        """`allreduce` after round 0 (`c_idx`: this core's index on its chip)."""
        del first
        cp, scr, me, sc = self.cp, self.scr, self.me, self.sc
        HIER, RSAG, R, HC2 = cp.HIER, cp.RSAG, cp.R, cp.HC2
        send, recv = scr["send"], scr["recv"]
        if HIER == 3:
            for rnd_i in range(1, cp.N_HIER3):
                with sc("ar_sum"):
                    bi = lax.rem(me >> rnd_i, 2)
                    send[rnd_i, slot] = (
                        recv[rnd_i - 1, slot, 0].astype(F32)
                        + recv[rnd_i - 1, slot, 1].astype(F32)
                    ).astype(BF16)
                    recv[rnd_i, slot, bi] = send[rnd_i, slot]
                with sc("ar_comm"):
                    self.ar_round(slot, rnd_i, fill if rnd_i == 1 else None)
            with sc("ar_sum"):
                last = cp.N_HIER3 - 1
                return recv[last, slot, 0].astype(F32) + recv[last, slot, 1].astype(F32)
        if HIER in (1, 2):
            ag_send, ag_recv = scr["ag_send"], scr["ag_recv"]
            HX = HIER == 2 and cp.HX
            if HX:
                q_w = HC2 // 4
                q_lo = pl.multiple_of((me // 2) * q_w, q_w)
            with sc("ar_sum"):
                ag_send[slot] = (
                    recv[slot, 0].astype(F32) + recv[slot, 1].astype(F32)
                ).astype(BF16)
                if HX:  # this chip's own quarter of the half-sum
                    ag_recv[slot, me // 2, :, 0:q_w] = ag_send[
                        slot, :, pl.ds(q_lo, q_w)
                    ]
                else:
                    ag_recv[slot, me // 2] = ag_send[slot]
            if HIER == 1 and defer:
                assert cp.ARP and slot == 1, (cp.ARP, slot)
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
                self.ar_round(slot, 1, fill)
            if HIER == 1:
                with sc("ar_sum"):
                    return jnp.sum(ag_recv[slot].astype(F32), axis=0)
            s2_send, s2_recv = scr["s2_send"], scr["s2_recv"]
            if HX:
                with sc("ar_sum"):  # this chip's quarter over the 4 chips, in order
                    s2_send[slot, :, pl.ds(q_lo, q_w)] = jnp.sum(
                        ag_recv[slot, :, :, 0:q_w].astype(F32), axis=0
                    ).astype(BF16)
                with sc("ar_comm"):
                    self.ar_round(slot, 3)
                if not defer:
                    with sc("ar_sum"):
                        s2_recv[slot, c_idx] = s2_send[slot]
            else:
                with sc("ar_sum"):
                    s2_send[slot] = jnp.sum(ag_recv[slot].astype(F32), axis=0).astype(
                        BF16
                    )
                    if not defer:
                        s2_recv[slot, c_idx] = s2_send[slot]
            if defer:
                assert cp.ARP and slot == 1, (cp.ARP, slot)
                with sc("ar_comm"):
                    for d in self.ar_copies(slot, 2):
                        d.start()
                with sc("ar_res"):
                    self.arp_half_add(s2_send[slot].astype(F32), self.arp_c)
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
            self.ar_round(slot, 1, fill)
        with sc("ar_sum"):
            return jnp.concatenate(
                [ag_recv[slot, j] for j in range(R)], axis=-1
            ).astype(F32)

    def add_residual(self, tot):
        """Unfold `tot` if folded and add to `scr['res']` with BF16 rounding."""
        cp, scr, sc = self.cp, self.scr, self.sc
        res = scr["res"]
        with sc("ar_res"):
            if not cp.WAR:
                tot = jnp.concatenate(
                    [tot[j * cp.B : (j + 1) * cp.B] for j in range(cp.HPC)], axis=-1
                )
            res[0 : cp.B, :] = round_bf16(res[0 : cp.B, :] + tot[0 : cp.B])

    def drain_sends(self):
        """Wait for the last layer's all-reduce sends before kernel exit."""
        if self.comm:
            for slot in sorted(self._used_slots):
                for rnd_i in self.ar_rounds(slot):
                    for d in self.ar_copies(slot, rnd_i):
                        d.wait_send()
