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
"""MoE routing: router top-k, dedup into hit lists, first expert-ring fetches."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .context import MoEEmit
from .dedup import dedup_ecopy, dedup_local, dedup_vec
from .ring import fetch_e, fetch_loop, fetch_range
from .topk import route_topk, route_weights


def route(m: MoEEmit, l_idx: Any, p8: Any) -> Any:
    """Route layer `l_idx`'s rows; returns this rank's routed-expert count."""
    ctx, ex, mp, scr, sc, sp, skip = m.ctx, m.ex, m.mp, m.scr, m.sc, m.sp, m.skip
    MR, R = ex.MR, ex.R
    E, ECOPY, EXTRA, NBE, TE = mp.E, mp.ECOPY, mp.EXTRA, mp.NBE, mp.TE
    choice = p8
    if m.dbg.forced_routing:
        # `MoEModule.prologue` issues the initial `rbias[0] -> rb[0]` DMA;
        # `wait()` only consumes the destination shape and `rb_sem[rs]`.
        rs = l_idx % 2
        pltpu.make_async_copy(
            ctx.extra["rbias"].at[0], scr["rb"].at[rs], scr["rb_sem"].at[rs]
        ).wait()

        @pl.when(l_idx + 1 < sp.L)
        def _():
            pltpu.make_async_copy(
                ctx.extra["rbias"].at[l_idx + 1],
                scr["rb"].at[1 - rs],
                scr["rb_sem"].at[1 - rs],
            ).start()

        choice = scr["rb"][rs, 0:MR, :]
    # Note on ordering: when `VEC` (`route_vec=True`) is active,
    # `elist`/`wlist` are populated in ascending local slot order
    # (`0..ER-1`) and `cplist` is populated in ascending replica-slot
    # order (`0..NCP-1`). When `VEC` is False (`ecopy_opt` 0 or 1),
    # both lists are populated in token/pick first-appearance order
    # (`b=0..B-1, k=0..TOPK-1`). Because Po2C is greedy and expert
    # partial sums accumulate into `part` in `elist` order, `route_vec`
    # is a distinct routing/Po2C policy (the default `route_vec=False`
    # preserves first-appearance order).
    nb_picks = scr["elist"].shape[0]
    VEC = mp.route_vec
    slot_ec = l_idx % 2
    NCP = R * EXTRA
    MROW = E // 128
    # route_early: the routing weights inside the hit-list copy's round trip,
    # and (VEC) the first expert fills right after the fixed-hit scan.
    early = mp.route_early
    with sc("route_topk", 3):
        sel = route_topk(m, choice, p8, VEC, slot_ec, NCP, MROW, write_wn=not early)
    with sc("route_dma", 3):
        cp = pltpu.make_async_copy(scr["hitv"], scr["hits"], scr["misc_sem"])
        cp.start()
        if early:
            route_weights(m, sel, p8)
        cp.wait()
    # `MoEModule.prologue` / `init_buffers` zeroes `scr["seen"]` before layer 0
    # of every step, so `stamp = l_idx + 1` (`1..L`) is unique per layer.
    stamp = l_idx + 1
    n_tot: Any = jnp.int32(0)
    n_hit: Any = jnp.int32(0)
    n_cp: Any = jnp.int32(0)
    early_fill = early and VEC and "experts" not in skip
    fill0: list[Any] = []

    def first_fill(n_prim: Any) -> None:
        fill0.append(jnp.minimum(NBE, n_prim * TE))
        with sc("route_fetch0", 3):
            fetch_range(m, jnp.int32(0), fill0[0], l_idx, fixed=mp.prim_now)

    if VEC:
        n_tot, n_hit, n_cp = dedup_vec(
            m, NCP, MROW, nb_picks, early_fill=first_fill if early_fill else None
        )
    if ECOPY:
        n_hit = dedup_ecopy(m, VEC, slot_ec, stamp, n_tot, n_hit, n_cp)
    else:
        n_hit = dedup_local(m, stamp)
    if "experts" in skip:
        n_hit = jnp.int32(0)
    scr["st"][0] = n_hit * TE
    with sc("route_fetch", 3):
        if early_fill:
            fetch_range(m, fill0[0], jnp.minimum(NBE, n_hit * TE), l_idx)
        elif mp.stack_e:
            for k in range(NBE):

                @pl.when(k < n_hit * TE)
                def _(k=k):
                    fetch_e(m, k, l_idx)
        else:
            if mp.unroll and mp.group_rel:
                fetch_loop(m, jnp.minimum(NBE, n_hit * TE), l_idx)
            else:
                lax.fori_loop(
                    0,
                    jnp.minimum(NBE, n_hit * TE),
                    lambda k, _: (fetch_e(m, k, l_idx), 0)[1],
                    0,
                )
    return n_hit
