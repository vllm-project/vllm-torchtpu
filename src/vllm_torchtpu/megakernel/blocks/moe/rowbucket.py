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
"""Routed-row bucketing for the `PIPE_EP` expert loop (`MoETunables.row_bucket`).

The dense loop multiplies every fetched expert tile against all `MB` rows,
although an expert is routed to only a few of them. Here each expert runs on
its routed rows only:

- Per layer (`rb_tables`, after the router): `rbcs[e]` = routed-row count of
  expert `e` (SMEM, read as a scalar to pick the row block), and the row code
  `code[r, e]` = rank of row `r` among expert `e`'s routed rows, or -1
  (`rbpre` [MR, E] bf16 and its transpose `rbpreT` [E, 128] f32).
- Per expert: count <= `rb_small` rows -> one `rb_small`-row block; more ->
  `rb_chunk`-row chunks (a dynamic loop, so a hot expert costs rows, not a
  cliff). A block's rows are gathered from `xb` by a one-hot matmul
  (`P [G, MB] @ xb`, exact: one nonzero term per output), e13 / e2 run on
  the `G` rows, and the weighted rows are scattered back with the one-hot
  transpose (`PT [MR, G] @ y`, exact since `y` is bf16-rounded) and added
  into `part` on the block's rows only.

The weight tiles, their order, every `etake` / `egive` and every per-row
f32 operation are the same as in `experts_piped_ep`, so `part` is bitwise
equal to the dense loop's, up to the sign of a zero `part` entry on a row the
expert does not route (dense adds `0 * y`, which is -0.0 for y < 0; here +0.0).
On XLA:CPU (interpret mode) compare with `--xla_allow_excess_precision=false`
(and FMA off): otherwise XLA drops some `rnd` f32 -> bf16 -> f32 round trips,
depending on the graph around them (e.g. inside a `pl.when` branch), in
either loop (tests/megakernel/test_moe_rowbucket.py).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...ops import BF16, F32, I32, rnd, rup
from .context import MoEEmit
from .experts import _ABLATE, _mm
from .ring import egive, egive_group, es_wait, etake


def rb_scratch(ex: Any, mp: Any) -> dict[str, Any]:
    """Scratch of the row-bucketed expert loop (row codes are small exact ints)."""
    nrow = rup(mp.E // 128, 8)
    return {
        "rbpre": pltpu.VMEM((ex.MR, mp.E), BF16),
        "rbpreT": pltpu.VMEM((mp.E, 128), F32),  # f32: dynamic single-row loads
        "rbcv": pltpu.VMEM((nrow, 128), I32),
        "rbcs": pltpu.SMEM((nrow, 128), I32),
        "rbc_sem": pltpu.SemaphoreType.DMA(()),
    }


def rb_tables(m: MoEEmit) -> None:
    """Routed-row counts (`rbcv`, DMA'd to `rbcs` by `route`) and row codes from `wn`."""
    ex, mp, scr, stream = m.ex, m.mp, m.scr, m.stream
    MR, E = ex.MR, mp.E
    mk = jnp.where(scr["wn"][...] != 0.0, 1.0, 0.0)  # [MR, E] routed (row, expert)
    cnt = jnp.sum(mk, axis=0, keepdims=True).astype(I32)  # [1, E]
    nrow = scr["rbcv"].shape[0]
    for j in range(E // 128):
        scr["rbcv"][j : j + 1, :] = cnt[:, j * 128 : (j + 1) * 128]
    if E // 128 < nrow:
        scr["rbcv"][E // 128 :, :] = jnp.zeros((nrow - E // 128, 128), I32)
    r_i = lax.broadcasted_iota(I32, (MR, MR), 0)
    c_i = lax.broadcasted_iota(I32, (MR, MR), 1)
    lstrict = jnp.where(c_i < r_i, 1.0, 0.0).astype(BF16)
    # pre[r, e] = routed rows of expert e above row r (0/1 bf16 products, f32 sums: exact).
    pre = stream.mm(lstrict, mk.astype(BF16))
    code = jnp.where(mk > 0.0, pre, -1.0)  # [MR, E]
    scr["rbpre"][...] = code.astype(BF16)
    pad = 128 - MR  # transpose a full 128-row tile (as `topk_rows_on_lanes`)
    if pad:
        code = jnp.concatenate([code, jnp.full((pad, E), -1.0, F32)], axis=0)
    scr["rbpreT"][...] = code.T  # [E, 128]


def experts_piped_ep_rb(m: MoEEmit, l_idx, n_hit):
    """`experts_piped_ep` with each expert's matmuls on its routed rows only."""
    ex, lane_e8, mp, scr, stream, tk = m.ex, m.lane_e8, m.mp, m.scr, m.stream, m.tk
    MB, MR = ex.MB, ex.MR
    H, IR, NSS, T13, T2, TE = mp.H, mp.IR, mp.NSS, mp.T13, mp.T2, mp.TE
    G0, GC = mp.rb_small, mp.rb_chunk
    RBLK = 32  # scatter row block (bounds the [rows, H] temporaries)

    @pl.when(n_hit > 0)
    def _():
        acc, acc13 = scr["acc"], scr["acc13"]
        xb, xb2, part = scr["xb"], scr["xb2"], scr["part"]
        BN = 2 * IR
        TK13, TK2 = tk["e13"], tk["e2"]
        DIRECT0 = mp.r2_direct_store

        def count(e):
            return scr["rbcs"][lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)]

        def blocks(cnt, body):
            """`body(G, r0)` over the row blocks holding `cnt` gathered rows."""
            pl.when(cnt <= G0)(lambda: body(G0, 0))

            @pl.when(cnt > G0)
            def _():
                def blk(c, carry):
                    body(GC, pl.multiple_of(c * GC, GC))
                    return carry

                lax.fori_loop(0, (cnt + (GC - 1)) // GC, blk, 0)

        def gather_mat(e, G, r0):
            """One-hot `P [G, MB]`: P[j, r] = 1 iff row r is expert e's routed row r0 + j."""
            crow = scr["rbpreT"][pl.ds(e, 1), pl.ds(0, MB)]  # [1, MB]
            want = (lax.broadcasted_iota(I32, (G, MB), 0) + r0).astype(F32)
            return jnp.where(crow == want, 1.0, 0.0).astype(BF16)

        def code_col(e):
            """Row codes of expert e as a column `[MR, 1]` (f32 ints, -1 = not routed)."""
            return jnp.sum(
                jnp.where(lane_e8 == e, scr["rbpre"][...].astype(F32), 0.0),
                axis=1,
                keepdims=True,
            )

        def e13_tile(e, cnt, t, w13):
            k0 = t * TK13

            def body(G, r0):
                rows = pl.ds(r0, G)
                xg = stream.mm(gather_mat(e, G, r0), xb[:, k0 : k0 + TK13]).astype(BF16)
                mm13 = _mm(stream, xg, w13[...])
                if DIRECT0 and t == 0:
                    acc13[rows, :BN] = mm13
                else:
                    if t == 0:
                        acc13[rows, :BN] = jnp.zeros((G, BN), F32)
                    acc13[rows, :BN] += mm13

            blocks(cnt, body)

        def e2_tile(cnt, s_step, w2):
            k2 = s_step * TK2

            def body(G, r0):
                rows = pl.ds(r0, G)
                mm2 = _mm(stream, xb2[rows, k2 : k2 + TK2], w2[...])
                if DIRECT0 and s_step == 0:
                    acc[rows, :H] = mm2
                else:
                    if s_step == 0:
                        acc[rows, :H] = jnp.zeros((G, H), F32)
                    acc[rows, :H] += mm2

            blocks(cnt, body)

        def mid(cnt, s):
            if "mid" in _ABLATE:
                return

            def body(G, r0):
                rows = pl.ds(r0, G)
                h = acc13[rows, :BN] * scr["es"][s, :, :BN]
                mid_ = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:]))
                xb2[rows, :IR] = mid_.astype(BF16)

            blocks(cnt, body)

        def finish(e, cnt, s):
            """part[r] += wcol[r] * y[r] on expert e's routed rows (scatter of y)."""
            wcol = jnp.sum(
                jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True
            )
            ccol = code_col(e)
            if "acc" in _ABLATE:
                return

            def body(G, r0):
                y = rnd(acc[pl.ds(r0, G), :H] * scr["es"][s, :, BN:]).astype(BF16)
                want = (lax.broadcasted_iota(I32, (1, G), 1) + r0).astype(F32)
                for b0 in range(0, MR, RBLK):
                    nb = min(RBLK, MR - b0)
                    cc = ccol[b0 : b0 + nb]
                    pt = jnp.where(cc == want, 1.0, 0.0).astype(BF16)  # [nb, G]
                    ys = stream.mm(pt, y)  # exact scatter of the bf16 rows
                    # The dense loop's `part += wcol * y`; rows outside the block add
                    # wcol * 0 = 0.
                    part[b0 : b0 + nb, :] += wcol[b0 : b0 + nb] * ys

            blocks(cnt, body)

        def rel(items, l_idx):
            if mp.group_rel:
                egive_group(m, items, l_idx)
            else:
                for u, t in items:
                    egive(m, u, l_idx, t)

        # Expert 0's gate/up.
        e0 = scr["elist"][0]
        c0 = count(e0)
        for t in range(T13):
            w13 = etake(m, "e13", t)
            e13_tile(e0, c0, t, w13)
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(t - 1, t - 1), (t, t)], l_idx)
                elif t == T13 - 1:
                    egive(m, t, l_idx, t)
            else:
                egive(m, t, l_idx, t)
        es_wait(m, 0)
        mid(c0, 0)

        def body_ep(i, _):
            e = scr["elist"][i]
            e_n = scr["elist"][i + 1]
            c, c_n = count(e), count(e_n)
            s_cur = i % NSS
            s_nxt = (i + 1) % NSS
            base_w2 = i * TE + T13
            base_nxt13 = (i + 1) * TE
            for s_step in range(T2):
                u2 = base_w2 + s_step
                t0, t1 = 2 * s_step, 2 * s_step + 1
                u13_0, u13_1 = base_nxt13 + t0, base_nxt13 + t1
                w2 = etake(m, "e2", u2)
                w13_0 = etake(m, "e13", u13_0)
                e2_tile(c, s_step, w2)
                e13_tile(e_n, c_n, t0, w13_0)
                rel([(u2, T13 + s_step), (u13_0, t0)], l_idx)
                w13_1 = etake(m, "e13", u13_1)
                e13_tile(e_n, c_n, t1, w13_1)
                egive(m, u13_1, l_idx, t1)
            finish(e, c, s_cur)
            es_wait(m, i + 1)
            mid(c_n, s_nxt)
            return 0

        lax.fori_loop(0, n_hit - 1, body_ep, 0)

        i_last = n_hit - 1
        e_last = scr["elist"][i_last]
        c_last = count(e_last)
        s_last = i_last % NSS
        base_last_w2 = i_last * TE + T13
        for s_step in range(T2):
            u2 = base_last_w2 + s_step
            w2 = etake(m, "e2", u2)
            e2_tile(c_last, s_step, w2)
            if mp.group_rel:
                if s_step % 2 == 1:
                    egive_group(
                        m, [(u2 - 1, T13 + s_step - 1), (u2, T13 + s_step)], l_idx
                    )
                elif s_step == T2 - 1:
                    egive(m, u2, l_idx, T13 + s_step)
            else:
                egive(m, u2, l_idx, T13 + s_step)
        finish(e_last, c_last, s_last)
