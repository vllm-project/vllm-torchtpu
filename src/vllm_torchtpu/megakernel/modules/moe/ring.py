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
"""MoE routed-expert ring: issue / take / release dynamic expert weight units.

A routed expert `i` is `TE = T13 + T2` weight units (`e13` tiles then `e2`
tiles); unit `u` streams into dynamic bank `u % NBE`, and its per-expert
scales into `es` slot `i % NSS`. `scr["st"][0]` holds the unit count of
the current layer, so a release only refills while units remain.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...ops import bswitch
from .context import MoEEmit


def fetch_e(m: MoEEmit, u, l_idx, t=None):
    """Issue routed expert unit `u` into dynamic bank `u % NBE`."""
    W, me, mp, scr, stream = m.W, m.me, m.mp, m.scr, m.stream
    ECOPY, ER, NBE, NSS, T13, TE = mp.ECOPY, mp.ER, mp.NBE, mp.NSS, mp.T13, mp.TE
    if t is None and isinstance(u, int):
        t = u % TE
    i = u // TE if t is None else (u - t) // TE
    if ECOPY:
        w = scr["wlist"][i]
    else:
        e = scr["elist"][i]
        w = e if mp.moe_tp else e - me * ER
    bank = u % NBE

    def es_copy():
        if "es" not in W:
            return
        s = i % NSS
        pltpu.make_async_copy(
            W["es"].at[l_idx, w], scr["es"].at[s], scr["es_sem"].at[s]
        ).start()

    def e13_start(tt):
        if mp.stack_e:
            stream.ecopy("e13", W["e13"].at[l_idx, w, tt], bank).start()
        else:
            bswitch(
                l_idx,
                [
                    lambda r=r: stream.ecopy("e13", r.at[w, tt], bank).start()
                    for r in W["e13"]
                ],
            )

    def e2_start(tt):
        if mp.stack_e:
            stream.ecopy("e2", W["e2"].at[l_idx, w, tt], bank).start()
        else:
            bswitch(
                l_idx,
                [
                    lambda r=r: stream.ecopy("e2", r.at[w, tt], bank).start()
                    for r in W["e2"]
                ],
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


def etake(m: MoEEmit, kind: str, u):
    W, mp, sc, stream = m.W, m.mp, m.sc, m.stream
    NBE = mp.NBE
    bank = u % NBE
    with sc("ewait", 3):
        w0 = W[kind].at[0, 0, 0] if mp.stack_e else W[kind][0].at[0, 0]
        stream.ecopy(kind, w0, bank).wait()
    return stream.eview(kind, bank)


def egive(m: MoEEmit, u, l_idx, t=None):
    mp, scr = m.mp, m.scr
    NBE, TE = mp.NBE, mp.TE
    nu = u + NBE
    nt_ = None if t is None else (t + NBE) % TE

    @pl.when(nu < scr["st"][0])
    def _():
        fetch_e(m, nu, l_idx, nt_)


def _unit_meta(m: MoEEmit, u, t):
    me, mp, scr = m.me, m.mp, m.scr
    ECOPY, ER, NBE, TE = mp.ECOPY, mp.ER, mp.NBE, mp.TE
    if t is None and isinstance(u, int):
        t = u % TE
    dyn = not isinstance(u, int)
    if t is None:
        i = lax.div(u, jnp.int32(TE)) if dyn else u // TE
    else:
        i = lax.div(u - t, jnp.int32(TE)) if dyn else (u - t) // TE
    lst = scr["wlist"] if ECOPY else scr["elist"]
    n_l = lst.shape[0]
    ic = i if isinstance(i, int) else jnp.minimum(i, n_l - 1)
    if ECOPY:
        w = lst[ic]
    else:
        e = lst[ic]
        w = e if mp.moe_tp else e - me * ER
    td = None if t is not None else u - i * TE
    bank = lax.rem(u, jnp.int32(NBE)) if dyn else u % NBE
    return i, w, bank, t, td


def _es_start(m: MoEEmit, i, w, l_idx):
    W, mp, scr = m.W, m.mp, m.scr
    if "es" not in W:
        return
    NSS = mp.NSS
    s = lax.rem(i, jnp.int32(NSS)) if not isinstance(i, int) else i % NSS
    pltpu.make_async_copy(
        W["es"].at[l_idx, w], scr["es"].at[s], scr["es_sem"].at[s]
    ).start()


def _arm_issue(m: MoEEmit, r13, r2, w, bank, t, td, gate):
    mp, stream = m.mp, m.stream
    T13 = mp.T13

    def g(p):
        return p if gate is None else jnp.logical_and(gate, p)

    if t is not None:

        def go():
            if t < T13:
                stream.ecopy("e13", r13.at[w, t], bank).start()
            else:
                stream.ecopy("e2", r2.at[w, t - T13], bank).start()

        if gate is None:
            go()
        else:
            pl.when(gate)(go)
        return

    @pl.when(g(td < T13))
    def _():
        stream.ecopy("e13", r13.at[w, td], bank).start()

    @pl.when(g(td >= T13))
    def _():
        stream.ecopy("e2", r2.at[w, td - T13], bank).start()


def egive_group(m: MoEEmit, items, l_idx):
    W, mp, scr = m.W, m.mp, m.scr
    NBE, TE = mp.NBE, mp.TE
    if mp.stack_e:
        for u, t in items:
            egive(m, u, l_idx, t)
        return
    st = scr["st"][0]
    metas = []
    for u, t in items:
        nu = u + NBE
        nt_ = None if t is None else (t + NBE) % TE
        metas.append(_unit_meta(m, nu, nt_) + (nu < st,))

    def arm(r13, r2):
        for _, w, bank, t, td, gate in metas:
            _arm_issue(m, r13, r2, w, bank, t, td, gate)

    bswitch(
        l_idx,
        [lambda a=a, b=b: arm(a, b) for a, b in zip(W["e13"], W["e2"])],
    )
    for i, w, _, t, td, gate in metas:
        if t is not None:
            if t == 0:
                pl.when(gate)(lambda i=i, w=w: _es_start(m, i, w, l_idx))
        else:
            pl.when(jnp.logical_and(gate, td == 0))(
                lambda i=i, w=w: _es_start(m, i, w, l_idx)
            )


def fetch_loop(m: MoEEmit, nf, l_idx):
    W = m.W

    def arm(r13, r2):
        def body(k, carry):
            i, w, _, t, td = _unit_meta(m, k, None)
            _arm_issue(m, r13, r2, w, k, t, td, None)
            pl.when(td == 0)(lambda: _es_start(m, i, w, l_idx))
            return carry

        lax.fori_loop(0, nf, body, 0)

    bswitch(
        l_idx,
        [lambda a=a, b=b: arm(a, b) for a, b in zip(W["e13"], W["e2"])],
    )


def es_wait(m: MoEEmit, i):
    W, mp, sc, scr = m.W, m.mp, m.sc, m.scr
    if "es" not in W:
        return
    NSS = mp.NSS
    s = i % NSS
    with sc("swait", 3):
        pltpu.make_async_copy(
            W["es"].at[0, 0], scr["es"].at[s], scr["es_sem"].at[s]
        ).wait()
