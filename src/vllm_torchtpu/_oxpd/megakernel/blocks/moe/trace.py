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
"""q38 X-K-LIVE (L3-LIVE): `MoEDebug.hit_trace`, an engagement trace of the routed-expert work (tests only).

Per layer and rank, after `route` (dedup + first expert-ring fetches), three DMAs copy what the routing decided:
  TRACE_HITS  [MR, 128] int32  `hitv`: row b's top-k picks in lanes 0..TOPK-1 (expert ids);
  TRACE_ELIST [nb_picks] int32 `elist`: this rank's routed experts in hit-list order (first n_hit entries valid);
  TRACE_UNITS [1] int32        `st` = n_hit * TE: the expert weight units this layer issues (`route_fetch` issues the
                               first min(NBE, st), each release refills while the unit index < st: one expert-weight DMA
                               per unit, `ring.fetch_e` / `egive`).
A test classifies every issued unit by the rows that picked its expert (live / pad-only / nobody), using the rows'
own picks, so the classification never trusts the mask under test. Off by default: no output, no DMA, no op.
"""

from __future__ import annotations

from typing import Any

from jax.experimental.pallas import tpu as pltpu

from ...ops import I32, rup
from ...spec import StateDecl

TRACE_HITS = "moe_trace_hits"
TRACE_ELIST = "moe_trace_elist"
TRACE_UNITS = "moe_trace_units"
TRACE_NAMES = (TRACE_HITS, TRACE_ELIST, TRACE_UNITS)


def trace_outputs(dbg: Any, g: Any, ex: Any) -> list[Any]:
    if not dbg.hit_trace:
        return []
    nb_picks = rup(ex.B * g.TOPK, 128)
    return [
        StateDecl(TRACE_HITS, (ex.MR, 128), I32, "w"),
        StateDecl(TRACE_ELIST, (nb_picks,), I32, "w"),
        StateDecl(TRACE_UNITS, (1,), I32, "w"),
    ]


def hit_trace(m: Any, l_idx: Any) -> None:
    """Copy `hitv`, `elist` and `st` of layer `l_idx` to the trace outputs (synchronous, on `misc_sem`)."""
    scr, O = m.scr, m.ctx.O  # noqa: E741
    o = m.sp.block_ord(l_idx, "moe")
    for src, name in ((scr["hitv"], TRACE_HITS), (scr["elist"], TRACE_ELIST), (scr["st"], TRACE_UNITS)):
        c = pltpu.make_async_copy(src, O[name].at[o], scr["misc_sem"])
        c.start()
        c.wait()
