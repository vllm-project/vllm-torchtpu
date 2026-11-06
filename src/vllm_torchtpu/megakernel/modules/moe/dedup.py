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
"""MoE routing dedup: per-layer hit lists, redundant-expert (Po2C) placement."""

from __future__ import annotations

import functools
from typing import Any

import jax.numpy as jnp
from jax import lax

from ...ops import I32
from .context import MoEEmit


def dedup_vec(m: MoEEmit, NCP: int, MROW: int, nb_picks: int) -> tuple[Any, Any, Any]:
    """Read the vectorized route tables back into `elist`/`wlist`/`cplist`."""
    scr, sc, me = m.scr, m.sc, m.me
    R, ER = m.ex.R, m.mp.ER
    with sc("route_dedup", 3):
        for r in range(R):
            scr["rload"][r] = scr["hits"][MROW, NCP + r]
        n_tot = scr["hits"][MROW, NCP + R]
        base = me * ER

        def prim(s, n):
            e = base + s
            f = scr["hits"][lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)]
            idx = jnp.minimum(n, nb_picks - 1)
            scr["elist"][idx] = e
            scr["wlist"][idx] = s
            return n + f

        n_hit = lax.fori_loop(0, ER, prim, jnp.int32(0), unroll=True)

        def cpl(j, n):
            v1 = scr["hits"][MROW, j]
            idx = jnp.minimum(n, nb_picks - 1)
            scr["cplist"][idx] = v1 - 1
            return n + jnp.where(v1 > 0, 1, 0).astype(I32)

        n_cp = lax.fori_loop(0, NCP, cpl, jnp.int32(0), unroll=True)
    return n_tot, n_hit, n_cp


def dedup_ecopy(
    m: MoEEmit,
    VEC: bool,
    slot_ec: Any,
    stamp: Any,
    n_tot: Any,
    n_hit: Any,
    n_cp: Any,
) -> Any:
    """Hit lists with redundant expert copies; returns this rank's hit count."""
    ex, mp, scr, sc, me = m.ex, m.mp, m.scr, m.sc, m.me
    B, R, WIDE = ex.B, ex.R, ex.WIDE
    ER, TOPK = mp.ER, mp.TOPK
    if not VEC:
        for r in range(R):
            scr["rload"][r] = jnp.int32(0)

    if VEC:
        pass
    elif mp.ecopy_opt == 1:

        def pick_uniq(b, k, n_tot):
            e = scr["hits"][b, k]
            hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
            new = scr["seen"][hi, lo] != stamp
            scr["seen"][hi, lo] = stamp
            scr["ulist"][n_tot] = e
            return n_tot + jnp.where(new, 1, 0).astype(I32)

        def classify_uniq(i, carry_c):
            n, n_cp = carry_c
            e = scr["ulist"][i]
            code = scr["ecopy"][slot_ec, 0, e].astype(I32)
            has_cp = code > 0
            r0 = lax.div(e, ER)
            s0 = e - r0 * ER
            scr["rload"][r0] = scr["rload"][r0] + jnp.int32(1)
            mine_fixed = jnp.logical_and(jnp.logical_not(has_cp), r0 == me)
            scr["elist"][n] = e
            scr["wlist"][n] = s0
            scr["cplist"][n_cp] = e
            return (
                n + jnp.where(mine_fixed, 1, 0).astype(I32),
                n_cp + jnp.where(has_cp, 1, 0).astype(I32),
            )

        n_tot = jnp.int32(0)
        with sc("route_dedup", 3):
            if WIDE:

                def row_uniq(b, nt_in):
                    return lax.fori_loop(
                        0,
                        TOPK,
                        functools.partial(pick_uniq, b),
                        nt_in,
                        unroll=True,
                    )

                n_tot = lax.fori_loop(0, B, row_uniq, n_tot, unroll=2)
            else:
                for b in range(B):
                    n_tot = lax.fori_loop(
                        0,
                        TOPK,
                        functools.partial(pick_uniq, b),
                        n_tot,
                        unroll=True,
                    )
            n_hit, n_cp = lax.fori_loop(
                0, n_tot, classify_uniq, (jnp.int32(0), jnp.int32(0))
            )
    else:

        def pick_ecopy(b, k, carry):
            n, n_cp, n_tot = carry
            e = scr["hits"][b, k]
            hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
            new = scr["seen"][hi, lo] != stamp
            scr["seen"][hi, lo] = stamp
            code = scr["ecopy"][slot_ec, 0, e].astype(I32)
            has_cp = code > 0
            r0 = lax.div(e, ER)
            s0 = e - r0 * ER
            fixed_new = jnp.logical_and(new, jnp.logical_not(has_cp))
            cp_new = jnp.logical_and(new, has_cp)
            scr["rload"][r0] = scr["rload"][r0] + jnp.where(new, 1, 0).astype(I32)
            mine_fixed = jnp.logical_and(fixed_new, r0 == me)
            scr["elist"][n] = e
            scr["wlist"][n] = s0
            scr["cplist"][n_cp] = e
            return (
                n + jnp.where(mine_fixed, 1, 0).astype(I32),
                n_cp + jnp.where(cp_new, 1, 0).astype(I32),
                n_tot + jnp.where(new, 1, 0).astype(I32),
            )

        carry = (jnp.int32(0), jnp.int32(0), jnp.int32(0))
        with sc("route_dedup", 3):
            if WIDE:

                def row_ecopy(b, c_in):
                    return lax.fori_loop(
                        0,
                        TOPK,
                        functools.partial(pick_ecopy, b),
                        c_in,
                        unroll=True,
                    )

                n_hit, n_cp, n_tot = lax.fori_loop(0, B, row_ecopy, carry, unroll=2)
            else:
                for b in range(B):
                    carry = lax.fori_loop(
                        0,
                        TOPK,
                        functools.partial(pick_ecopy, b),
                        carry,
                        unroll=True,
                    )
                n_hit, n_cp, n_tot = carry

    return po2c(m, slot_ec, n_tot, n_hit, n_cp)


def po2c(m: MoEEmit, slot_ec: Any, n_tot: Any, n_hit: Any, n_cp: Any) -> Any:
    """Power-of-two-choices placement of the experts with a redundant copy."""
    scr, sc, me = m.scr, m.sc, m.me
    R, ER, EXTRA = m.ex.R, m.mp.ER, m.mp.EXTRA
    with sc("route_dedup", 3):

        def po2c_pass1(j, _):
            e = scr["cplist"][j]
            r0 = lax.div(e, ER)
            c0 = scr["ecopy"][slot_ec, 0, e].astype(I32) - 1
            r1 = lax.div(c0, EXTRA)
            l0 = scr["rload"][r0]
            l1 = scr["rload"][r1]
            use_c = jnp.logical_or(
                l1 + 1 < l0,
                jnp.logical_and(
                    l1 < l0, jnp.logical_or(l0 * R > n_tot, l1 * R < n_tot)
                ),
            )
            d = jnp.where(use_c, 1, 0).astype(I32)
            scr["rload"][r0] = l0 - d
            scr["rload"][r1] = l1 + d
            scr["cpchoice"][j] = d
            return 0

        def po2c_pass2(j, n):
            e = scr["cplist"][j]
            r0 = lax.div(e, ER)
            s0 = e - r0 * ER
            c0 = scr["ecopy"][slot_ec, 0, e].astype(I32) - 1
            r1 = lax.div(c0, EXTRA)
            s1 = ER + (c0 - r1 * EXTRA)
            use_c = scr["cpchoice"][j] != 0
            cur = jnp.where(use_c, r1, r0)
            alt = jnp.where(use_c, r0, r1)
            l_cur = scr["rload"][cur]
            l_alt = scr["rload"][alt]
            flip = jnp.logical_or(
                l_alt + 1 < l_cur,
                jnp.logical_and(
                    l_alt < l_cur,
                    jnp.logical_and(l_cur * R > n_tot, l_alt * R < n_tot),
                ),
            )
            d = jnp.where(flip, 1, 0).astype(I32)
            scr["rload"][cur] = l_cur - d
            scr["rload"][alt] = l_alt + d
            final_c = jnp.not_equal(use_c, flip)
            pick_r = jnp.where(final_c, r1, r0)
            pick_w = jnp.where(final_c, s1, s0)
            mine = pick_r == me
            scr["elist"][n] = e
            scr["wlist"][n] = pick_w
            return n + jnp.where(mine, 1, 0).astype(I32)

        lax.fori_loop(0, n_cp, po2c_pass1, 0)
        n_hit = lax.fori_loop(0, n_cp, po2c_pass2, n_hit)
    return n_hit


def dedup_local(m: MoEEmit, stamp: Any) -> Any:
    """Hit list without redundant copies (TP: all experts; EP: this rank's)."""
    ctx, ex, mp, scr, sc, me = m.ctx, m.ex, m.mp, m.scr, m.sc, m.me
    B, WIDE = ex.B, ex.WIDE
    ER, TOPK = mp.ER, mp.TOPK

    def pick(b, k, n):
        e = scr["hits"][b, k]
        hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
        seen = scr["seen"][hi, lo]
        new = seen != stamp
        if not mp.moe_tp:
            new = jnp.logical_and(new, lax.div(e, ER) == me)
        if m.dbg.row_mask:
            on = ctx.extra["moe_rows"][b] != 0
            new = jnp.logical_and(new, on)
            scr["seen"][hi, lo] = jnp.where(on, stamp, seen)
        else:
            scr["seen"][hi, lo] = stamp
        scr["elist"][n] = e
        return n + jnp.where(new, 1, 0).astype(I32)

    n_hit = jnp.int32(0)
    with sc("route_dedup", 3):
        if WIDE:

            def row(b, n):
                return lax.fori_loop(
                    0, TOPK, functools.partial(pick, b), n, unroll=True
                )

            n_hit = lax.fori_loop(0, B, row, n_hit, unroll=2)
        else:
            for b in range(B):
                n_hit = lax.fori_loop(
                    0, TOPK, functools.partial(pick, b), n_hit, unroll=True
                )
    return n_hit
