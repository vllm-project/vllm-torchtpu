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
"""All-reduce of a sublayer-width `[MB, h]` partial for hyper-connection blocks.

The core ring reduces `[MB, g.H]`, the full residual width. A hyper-connection
sublayer produces an `h = g.H / NHC` wide partial, so its blocks share one
`AllReduceRing` instance built over `h` with its own `hc_`-prefixed buffers.
The payload is always unfolded (`[MR, h]`).
"""

from __future__ import annotations

from typing import Any

from jax.experimental.pallas import tpu as pltpu

from ...comm import AllReduceRing
from ...exec_spec import ExecSpec
from ...ops import BF16
from ...spec import ScratchReq
from ...vmem import CommParams

_RING_ATTR = "_hc_ring"
_DRAINED_ATTR = "_hc_ring_drained"
_BUF_NAMES = (
    "send",
    "recv",
    "ar_ssem",
    "ar_rsem",
    "ag_send",
    "ag_recv",
    "s2_send",
    "s2_recv",
)


def hc_comm_params(ex: ExecSpec, h: int) -> CommParams:
    """`CommParams` of an unfolded `[MR, h]` all-reduce over `ex.R` ranks."""
    R = ex.R
    tiles = ex.resolved_tiles
    hier = tiles.hier if R > 1 else 0
    if hier in (1, 2) and R != 8:
        hier = 0
    if hier == 2 and (h // 2) % 128 != 0:
        hier = 1
    n_hier3 = max(1, R.bit_length() - 1)
    rounds = (
        tuple(range(n_hier3))
        if hier == 3
        else ((0, 1, 2) if hier == 2 else ((0, 1) if hier == 1 else (0,)))
    )
    peers = 1 if hier == 3 else (3 if hier in (1, 2) else max(R - 1, 1))
    return CommParams(
        R=R,
        B=ex.B,
        M=ex.M,
        MR=ex.MR,
        MB=ex.MB,
        H=h,
        PR=ex.MR,
        PC=h,
        HPC=1,
        HIER=hier,
        RSAG=False,
        WAR=True,
        AR_R=ex.MR,
        AR_C=h,
        CW=h // R,
        HC2=h // 2,
        AR_ROUNDS=rounds,
        AR_PEERS=peers,
        N_HIER3=n_hier3,
    )


def hc_comm_scratch(ex: ExecSpec, h: int) -> list[ScratchReq]:
    """Persistent `hc_*` buffers of the shared sublayer all-reduce ring."""
    cp = hc_comm_params(ex, h)
    R, AR_R, AR_C, HC2 = cp.R, cp.AR_R, cp.AR_C, cp.HC2
    n_rounds, peers = len(cp.AR_ROUNDS), cp.AR_PEERS
    if cp.HIER == 3:
        send = (cp.N_HIER3, 2, AR_R, AR_C)
        recv = (cp.N_HIER3, 2, 2, AR_R, AR_C)
    elif cp.HIER == 2:
        send, recv = (2, 2, AR_R, HC2), (2, 2, AR_R, HC2)
    elif cp.HIER == 1:
        send, recv = (2, AR_R, AR_C), (2, 2, AR_R, AR_C)
    else:
        send, recv = (2, AR_R, AR_C), (2, R, AR_R, AR_C)
    reqs = [
        ScratchReq("hc_send", pltpu.VMEM(send, BF16)),
        ScratchReq("hc_recv", pltpu.VMEM(recv, BF16)),
        ScratchReq("hc_ar_ssem", pltpu.SemaphoreType.DMA((n_rounds, 2, peers))),
        ScratchReq("hc_ar_rsem", pltpu.SemaphoreType.DMA((n_rounds, 2, peers))),
    ]
    if cp.HIER == 1:
        reqs += [
            ScratchReq("hc_ag_send", pltpu.VMEM((2, AR_R, AR_C), BF16)),
            ScratchReq("hc_ag_recv", pltpu.VMEM((2, 4, AR_R, AR_C), BF16)),
        ]
    elif cp.HIER == 2:
        reqs += [
            ScratchReq("hc_ag_send", pltpu.VMEM((2, AR_R, HC2), BF16)),
            ScratchReq("hc_ag_recv", pltpu.VMEM((2, 4, AR_R, HC2), BF16)),
            ScratchReq("hc_s2_send", pltpu.VMEM((2, AR_R, HC2), BF16)),
            ScratchReq("hc_s2_recv", pltpu.VMEM((2, 2, AR_R, HC2), BF16)),
        ]
    return reqs


def hc_ring(ctx: Any, h: int) -> AllReduceRing:
    """The kernel-wide sublayer ring, created on first use and cached on `ctx`."""
    ring = getattr(ctx, _RING_ATTR, None)
    if ring is None:
        scr = {n: ctx.scr["hc_" + n] for n in _BUF_NAMES if "hc_" + n in ctx.scr}
        ring = AllReduceRing(
            hc_comm_params(ctx.ex, h), scr, ctx.me, skip=ctx.skip, sc=ctx.sc
        )
        setattr(ctx, _RING_ATTR, ring)
    return ring


def hc_drain(ctx: Any) -> None:
    """Wait for the ring's outstanding sends once per kernel."""
    ring = getattr(ctx, _RING_ATTR, None)
    if ring is None or getattr(ctx, _DRAINED_ATTR, False):
        return
    ring.drain_sends()
    setattr(ctx, _DRAINED_ATTR, True)
