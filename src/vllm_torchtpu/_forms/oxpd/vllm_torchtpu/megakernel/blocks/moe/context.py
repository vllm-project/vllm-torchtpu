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
"""Per-trace emit state shared by the `MoEBlock` emit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import lax

from ...ops import F32, I32
from .params import MoEDebug, MoEParams


@dataclass(frozen=True)
class MoEEmit:
    """References and constants one `MoEBlock.emit` trace reads.

    `lane_e8` / `lane128` / `row8` are the iotas the router and the expert
    weighting compare against; `EM` is the expert-id mask packed into the
    low mantissa bits of the top-k keys.
    """

    ctx: Any
    ex: Any
    mp: MoEParams
    dbg: MoEDebug
    W: Any
    scr: Any
    stream: Any
    me: Any
    skip: Any
    sc: Any
    sp: Any
    tk: Any
    lane_e8: Any
    lane128: Any
    row8: Any
    EM: int
    FLT_MIN: float

    @classmethod
    def build(cls, ctx: Any, mp: MoEParams, dbg: MoEDebug) -> MoEEmit:
        """Collect `ctx`'s refs and trace the shared iotas (in emit order)."""
        ex = ctx.ex
        stream = ctx.stream
        MR, E = ex.MR, mp.E
        lane_e8 = lax.broadcasted_iota(I32, (MR, E), 1)
        lane128 = lax.broadcasted_iota(I32, (MR, 128), 1)
        row8 = lax.broadcasted_iota(I32, (MR, 1), 0)
        EM = (1 << max(1, (E - 1).bit_length())) - 1
        FLT_MIN = float(jnp.finfo(F32).tiny)
        return cls(
            ctx=ctx,
            ex=ex,
            mp=mp,
            dbg=dbg,
            W=ctx.W,
            scr=ctx.scr,
            stream=stream,
            me=ctx.me,
            skip=ctx.skip,
            sc=ctx.sc,
            sp=stream.sp,
            tk=stream.sp.tk,
            lane_e8=lane_e8,
            lane128=lane128,
            row8=row8,
            EM=EM,
            FLT_MIN=FLT_MIN,
        )
