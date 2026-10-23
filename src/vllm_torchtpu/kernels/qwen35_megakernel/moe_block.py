"""MoE router, top-k / dedup / Po2C, shared expert, and routed expert pipelines."""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .config import (
    BF16,
    F32,
    I32,
    NT,
    TOPK_LANES,
    rnd,
    topk_rows_on_lanes,
)
from . import config as _cfg
from .context import MKBuildContext


class MoEBlock:
    """MoE router, top-k / dedup / Po2C, shared expert, and routed expert pipelines."""

    def __init__(self, ctx: MKBuildContext, rings):
        self.ctx = ctx
        self.rings = rings

    def es_wait(self, i):
        ctx = self.ctx
        W, scr, sc, NSS = ctx.W, ctx.scr, ctx.sc, ctx.NSS
        s = i % NSS
        with sc("swait", 2):
            pltpu.make_async_copy(
                W["es"].at[0, 0], scr["es"].at[s], scr["es_sem"].at[s]
            ).wait()

    def moe(self, l, slot, g0, is_att, defer_flush: bool = False):
        ctx = self.ctx
        c, scr, sc, sm, mm, rms_zc = (
            ctx.c,
            ctx.scr,
            ctx.sc,
            ctx.sm,
            ctx.mm,
            ctx.rms_zc,
        )
        dmatmul, dtake, flush_rel = (
            self.rings.dmatmul,
            self.rings.dtake,
            self.rings.flush_rel,
        )
        E, H, MR, MB = ctx.E, ctx.H, ctx.MR, ctx.MB
        AE, N_MOE, nt, tk, skip = ctx.AE, ctx.N_MOE, ctx.nt, ctx.tk, ctx.skip
        LAZY_MOE, PIPE, PIPE_EP = ctx.LAZY_MOE, ctx.PIPE, ctx.PIPE_EP

        seg_r = g0 + jnp.where(
            is_att,
            nt["att_in"] + nt["att_out"],
            nt["lin_in"] + nt["lin_out"],
        )

        def moe_flush():
            with sc("refill"):
                for lt_s in range(AE):
                    # MoE tiles start at in-layer offset base (static per layer type)
                    base = (
                        nt["att_in"] + nt["att_out"]
                        if lt_s == AE - 1
                        else nt["lin_in"] + nt["lin_out"]
                    )
                    pl.when(l % AE == lt_s)(
                        lambda lt_s=lt_s, base=base: flush_rel(
                            g0 + base, l // AE, lt_s, base, N_MOE
                        )
                    )

        SH_PRE = LAZY_MOE and "dense" not in skip and nt["sh13"] + nt["sh2"] <= 4
        with sc("moe_router"):
            x = rms_zc(scr["res"][...], sm(slot, "post_norm"))
            scr["xb"][...] = x.astype(BF16)
            if SH_PRE and getattr(ctx, "DWAIT_HOIST", False):
                n_r, n_13, n_2 = nt["router"], nt["sh13"], nt["sh2"]
                wg = self.rings.dtake_group(
                    [("router", seg_r + t) for t in range(n_r)]
                    + [("sh13", seg_r + n_r + t) for t in range(n_13)]
                    + [("sh2", seg_r + n_r + n_13 + t) for t in range(n_2)]
                )
                w13s, w2s = wg[n_r : n_r + n_13], wg[n_r + n_13 :]
                logits = dmatmul(
                    "router", seg_r, scr["xb"], E, lazy=LAZY_MOE, ws_pre=wg[:n_r]
                )[0:MR]
            else:
                logits = dmatmul("router", seg_r, scr["xb"], E, lazy=LAZY_MOE)[0:MR]
                if SH_PRE and not getattr(ctx, "DWAIT_HOIST", False):
                    g_s = seg_r + nt["router"]
                    w13s = [dtake("sh13", g_s + t) for t in range(nt["sh13"])]
                    w2s = [dtake("sh2", g_s + nt["sh13"] + t) for t in range(nt["sh2"])]
            glog = jnp.sum(x * sm(slot, "sh_gate"), axis=-1, keepdims=True)
            scr["sg"][...] = jnp.broadcast_to(jax.nn.sigmoid(glog), (MB, 128))
            mx = jnp.max(logits, axis=-1, keepdims=True)
            pexp = jnp.exp(logits - mx)
        SH_AT = ctx.SH_AT if SH_PRE else 0

        def shared_all():
            # SH_AT 1 / 2: the whole shared expert (sh13 then sh2), same ops in the same order
            with sc("moe_shared"):
                t13 = tk["sh13"]
                g13 = None
                for t, w in enumerate(w13s):
                    p_ = mm(scr["xb"][:, t * t13 : (t + 1) * t13], w[...])
                    g13 = p_ if g13 is None else g13 + p_
                g13 = g13 * sm(slot, "sh13_s")
                mid = rnd(
                    rnd(jax.nn.silu(rnd(g13[:, : c.ISR]))) * rnd(g13[:, c.ISR :])
                )
                scr["xb2"][:, : c.ISR] = mid.astype(BF16)
                t2 = tk["sh2"]
                ysh = None
                for t, w in enumerate(w2s):
                    p_ = mm(scr["xb2"][:, t * t2 : (t + 1) * t2], w[...])
                    ysh = p_ if ysh is None else ysh + p_
                scr["part"][...] = ysh * sm(slot, "sh2_s") * scr["sg"][:, 0:1]

        if SH_PRE and SH_AT == 0:
            with sc("moe_shared"):
                t13 = tk["sh13"]
                g13 = None
                for t, w in enumerate(w13s):
                    p_ = mm(scr["xb"][:, t * t13 : (t + 1) * t13], w[...])
                    g13 = p_ if g13 is None else g13 + p_
                g13 = g13 * sm(slot, "sh13_s")
                mid = rnd(
                    rnd(jax.nn.silu(rnd(g13[:, : c.ISR]))) * rnd(g13[:, c.ISR :])
                )
                scr["xb2"][:, : c.ISR] = mid.astype(BF16)
                if ctx.SH2_PRE_ROUTE:
                    t2 = tk["sh2"]
                    ysh = None
                    for t, w in enumerate(w2s):
                        p_ = mm(scr["xb2"][:, t * t2 : (t + 1) * t2], w[...])
                        ysh = p_ if ysh is None else ysh + p_
                    scr["part"][...] = ysh * sm(slot, "sh2_s") * scr["sg"][:, 0:1]
        with sc("moe_route"):
            n_hit = self.route(l, pexp, during_dma=shared_all if SH_AT == 1 else None)
        if SH_AT == 2:
            shared_all()
        if SH_PRE and SH_AT == 0 and not ctx.SH2_PRE_ROUTE:
            with sc("moe_shared"):
                t2 = tk["sh2"]
                ysh = None
                for t, w in enumerate(w2s):
                    p_ = mm(scr["xb2"][:, t * t2 : (t + 1) * t2], w[...])
                    ysh = p_ if ysh is None else ysh + p_
                scr["part"][...] = ysh * sm(slot, "sh2_s") * scr["sg"][:, 0:1]
        elif not SH_PRE:
            with sc("moe_shared"):
                # shared expert (TP slice of the intermediate), under the first expert tiles
                g13 = dmatmul(
                    "sh13", seg_r + nt["router"], scr["xb"], 2 * c.ISR, lazy=LAZY_MOE
                )
                g13 = g13 * sm(slot, "sh13_s")
                mid = rnd(rnd(jax.nn.silu(rnd(g13[:, :c.ISR]))) * rnd(g13[:, c.ISR:]))
                scr["xb2"][:, :c.ISR] = mid.astype(BF16)
                ysh = dmatmul(
                    "sh2",
                    seg_r + nt["router"] + nt["sh13"],
                    scr["xb2"],
                    H,
                    lazy=LAZY_MOE,
                )
                scr["part"][...] = ysh * sm(slot, "sh2_s") * scr["sg"][:, 0:1]
        with sc("moe_experts"):
            if PIPE:
                self.experts_piped(l, n_hit)
            elif PIPE_EP:
                self.experts_piped_ep(
                    l,
                    n_hit,
                    pre_drain=moe_flush if (LAZY_MOE and defer_flush) else None,
                )
            else:
                lax.fori_loop(0, n_hit, functools.partial(self.expert, l), 0)
        fill = None
        if LAZY_MOE:
            if not defer_flush:
                moe_flush()
            elif PIPE_EP and not PIPE:
                pl.when(n_hit == 0)(moe_flush)
            elif ctx.FLUSH_IN_AR1:
                fill = moe_flush
            else:
                moe_flush()
        if defer_flush:
            return scr["part"][...], fill
        return scr["part"][...]

    __call__ = moe

    def route(self, l, p8, during_dma=None):
        """Top-k of rows 0..MR-1 (live rows 0..B-1; p8 = their unnormalized probabilities) ->
        renormalized weights scr["wn"]; the deduplicated hit experts (EP: this rank's) ->
        scr["elist"][:n_hit] in pick order, and their first tiles start. Returns n_hit."""
        ctx = self.ctx
        c, tiles, scr, sc, sm, mm = ctx.c, ctx.tiles, ctx.scr, ctx.sc, ctx.sm, ctx.mm
        fetch_e, fetch_loop = self.rings.fetch_e, self.rings.fetch_loop
        B, L, E, R, ER, EXTRA = ctx.B, ctx.L, ctx.E, ctx.R, ctx.ER, ctx.EXTRA
        MR, NBE, TE, WIDE, ECOPY = ctx.MR, ctx.NBE, ctx.TE, ctx.WIDE, ctx.ECOPY
        EM, FLT_MIN = ctx.EM, ctx.FLT_MIN
        me, lane_e8, lane128, row8 = ctx.me, ctx.lane_e8, ctx.lane128, ctx.row8
        forced_routing, row_mask, skip = ctx.forced_routing, ctx.row_mask, ctx.skip
        rb_hbm, mr_ref = ctx.rb_hbm, ctx.mr_ref

        choice = p8
        if forced_routing:
            rs = l % 2
            pltpu.make_async_copy(
                rb_hbm.at[0], scr["rb"].at[rs], scr["rb_sem"].at[rs]
            ).wait()

            @pl.when(l + 1 < L)
            def _():
                pltpu.make_async_copy(
                    rb_hbm.at[l + 1],
                    scr["rb"].at[1 - rs],
                    scr["rb_sem"].at[1 - rs],
                ).start()

            choice = scr["rb"][rs, 0:MR, :]
        # Vector dedup (ECOPY): the hit set, per-rank loads and the hit copy candidates come
        # from the [MR, E] selection mask on the VPU/MXU; the scalar unit then only scans
        # this rank's ER primary slots and the R * EXTRA copy slots (not all B * TOPK picks).
        VEC = (
            tiles.route_vec
            and ECOPY
            and E % 128 == 0
            and E // 128 + 1 <= MR
            and R * EXTRA + R + 1 <= 128
        )
        slot = l % 2
        with sc("route_topk", 2):
            ids = jnp.zeros((MR, 128), I32)
            if tiles.topk:
                # one max-reduction per pick: the key keeps choice's high bits and holds
                # EM - lane in the low log2(E) mantissa bits, so keys are distinct and the
                # max names its lane. Ties (and values equal in all but those bits, 2^-14
                # relative for E=512) go to the lowest lane. choice >= 0; the FLT_MIN clamp
                # keeps keys normal floats (no denormal flushing).
                kb = lax.bitcast_convert_type(jnp.maximum(choice, FLT_MIN), I32)
                key = lax.bitcast_convert_type((kb & ~EM) | (EM - lane_e8), F32)
                if ctx.topk_lanes and E % 128 == 0 and MR <= 128 and c.TOPK <= 128:
                    # same keys, same picks in the same order (exact): no serial
                    # cross-lane reductions (see topk_rows_on_lanes)
                    ids_l, sel = topk_rows_on_lanes(key, c.TOPK, EM, need_ids=not VEC)
                    if not VEC:
                        ids = ids_l
                else:
                    for k in range(c.TOPK):
                        kmax = jnp.max(key, axis=-1, keepdims=True)
                        if not VEC:
                            am_ = EM - (lax.bitcast_convert_type(kmax, I32) & EM)
                            ids = jnp.where(lane128 == k, am_, ids)
                        key = jnp.where(key == kmax, -1.0, key)
                    sel = key < 0.0
            else:
                sel = jnp.zeros((MR, E), jnp.bool_)
                for k in range(c.TOPK):
                    cmax = jnp.max(choice, axis=-1, keepdims=True)
                    am_ = jnp.min(
                        jnp.where(choice == cmax, lane_e8, E), axis=-1, keepdims=True
                    )
                    ids = jnp.where(lane128 == k, am_, ids)
                    hit = lane_e8 == am_
                    sel = jnp.logical_or(sel, hit)
                    choice = jnp.where(hit, -jnp.inf, choice)
            if VEC:
                NCP = R * EXTRA
                MROW = E // 128
                hitf = jnp.max(
                    jnp.where(jnp.logical_and(sel, row8 < B), 1, 0).astype(I32),
                    axis=0,
                    keepdims=True,
                )  # [1, E]
                cpi = sm(slot, "ecopy").astype(I32)  # [1, E]: 0 or 1 + copy slot
                fixed = jnp.where(cpi > 0, 0, hitf)  # hit, no redundant copy
                for j in range(MROW):
                    scr["hitv"][j:j + 1, :] = fixed[:, j * 128:(j + 1) * 128]
                lane_1e = lax.broadcasted_iota(I32, (1, E), 1)
                v1 = (lane_1e + 1) * hitf
                row_a = lax.broadcasted_iota(I32, (16, E), 0)
                a = jnp.where(
                    row_a == 0,
                    lax.shift_right_logical(v1, 5),
                    jnp.where(
                        row_a == 1,
                        lax.bitwise_and(v1, 31),
                        jnp.where(row_a == 2, hitf, 0),
                    ),
                )
                a = a.astype(F32).astype(BF16)
                row_b = lax.broadcasted_iota(I32, (128, E), 0)
                lane_b = lax.broadcasted_iota(I32, (128, E), 1)
                rr = row_b - NCP
                in_rank = jnp.logical_and(lane_b >= rr * ER, lane_b < (rr + 1) * ER)
                cp_code = (row_b % R) * EXTRA + lax.div(row_b, R) + 1
                bm = jnp.where(
                    row_b < NCP,
                    jnp.where(cpi == cp_code, 1.0, 0.0),
                    jnp.where(
                        rr < R,
                        jnp.where(in_rank, 1.0, 0.0),
                        jnp.where(rr == R, 1.0, 0.0),
                    ),
                )
                bm = bm.astype(BF16)
                res = mm(a, bm, NT)  # [16, 128]
                lane_r = lax.broadcasted_iota(I32, (1, 128), 1)
                meta = jnp.where(lane_r < NCP, res[0:1] * 32.0 + res[1:2], res[2:3])
                scr["hitv"][MROW:MROW + 1, :] = meta.astype(I32)
            else:
                scr["hitv"][...] = ids
        with sc("route_dma", 2):
            cp = pltpu.make_async_copy(scr["hitv"], scr["hits"], scr["misc_sem"])
            cp.start()
            wsel = jnp.where(sel, p8, 0.0)
            scr["wn"][...] = jnp.where(
                row8 < B, wsel / jnp.sum(wsel, axis=-1, keepdims=True), 0.0
            )
            if during_dma is not None:  # SH_AT == 1: shared expert under the copy's round trip
                during_dma()
            cp.wait()
        stamp = l + 1
        n_fill0 = jnp.int32(0)
        RSHRINK = (
            getattr(ctx, "ROUTE_SHRINK", False)
            and VEC
            and E < (1 << getattr(ctx, "CP_SH", 16))
        )
        PRIM_NOW_K = getattr(ctx, "PRIM_NOW_K", False)

        def cp_dec(w):
            """Packed cplist entry -> (e, r0, s0, r1, s1): primary rank / slot, copy rank /
            slot (preserving Att15 cp_code mapping)."""
            CP_SH = getattr(ctx, "CP_SH", 16)
            e = lax.bitwise_and(w, (1 << CP_SH) - 1)
            j = lax.shift_right_logical(w, CP_SH)
            if ER & (ER - 1) == 0:
                r0 = lax.shift_right_logical(e, ER.bit_length() - 1)
                s0 = lax.bitwise_and(e, ER - 1)
            else:
                r0 = lax.div(e, ER)
                s0 = e - r0 * ER
            if R & (R - 1) == 0:
                r1 = lax.bitwise_and(j, R - 1)
                s1 = ER + lax.shift_right_logical(j, R.bit_length() - 1)
            else:
                r1 = lax.rem(j, R)
                s1 = ER + lax.div(j, R)
            return e, r0, s0, r1, s1

        if VEC:
            with sc("route_dedup", 2):
                for r in range(R):
                    scr["rload"][r] = scr["hits"][MROW, NCP + r]
                n_tot = scr["hits"][MROW, NCP + R]
                base = me * ER

                if PRIM_NOW_K:
                    def prim(s, n):
                        e = base + s
                        f = scr["hits"][
                            lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                        ]
                        scr["elist"][n] = e
                        return n + f
                else:
                    def prim(s, n):
                        e = base + s
                        f = scr["hits"][
                            lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                        ]
                        scr["elist"][n] = e
                        scr["wlist"][n] = s
                        return n + f

                n_hit = lax.fori_loop(0, ER, prim, jnp.int32(0), unroll=True)
                if PRIM_NOW_K:
                    scr["st"][5] = n_hit

                EARLY_FILL = VEC and "experts" not in skip
                if EARLY_FILL:
                    n_fill0 = jnp.minimum(NBE, n_hit * TE)
                    with sc("route_fetch0", 2):
                        self.rings.fetch_range(jnp.int32(0), n_fill0, l, fixed=PRIM_NOW_K)

                if RSHRINK:
                    CP_SH = getattr(ctx, "CP_SH", 16)
                    n_cp = jnp.int32(0)
                    for c0 in range(0, NCP, 16):
                        vs = [scr["hits"][MROW, j] for j in range(c0, min(c0 + 16, NCP))]
                        for k, v1 in enumerate(vs):
                            scr["cplist"][n_cp] = v1 + (((c0 + k) << CP_SH) - 1)
                            n_cp = n_cp + lax.shift_right_logical(-v1, 31)
                else:
                    def cpl(j, n):
                        v1 = scr["hits"][MROW, j]
                        scr["cplist"][n] = v1 - 1
                        return n + jnp.where(v1 > 0, 1, 0).astype(I32)

                    n_cp = lax.fori_loop(0, NCP, cpl, jnp.int32(0), unroll=True)
        if ECOPY:
            if not VEC:
                for r in range(R):
                    scr["rload"][r] = jnp.int32(0)

            if VEC:
                pass
            elif tiles.ecopy_opt == 1:
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
                    code = scr["ecopy"][slot, 0, e].astype(I32)
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
                with sc("route_dedup", 2):
                    if WIDE:
                        def row_uniq(b, nt_in):
                            return lax.fori_loop(
                                0,
                                c.TOPK,
                                functools.partial(pick_uniq, b),
                                nt_in,
                                unroll=True,
                            )

                        n_tot = lax.fori_loop(0, B, row_uniq, n_tot, unroll=2)
                    else:
                        for b in range(B):
                            n_tot = lax.fori_loop(
                                0,
                                c.TOPK,
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
                    code = scr["ecopy"][slot, 0, e].astype(I32)
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
                with sc("route_dedup", 2):
                    if WIDE:
                        def row_ecopy(b, c_in):
                            return lax.fori_loop(
                                0,
                                c.TOPK,
                                functools.partial(pick_ecopy, b),
                                c_in,
                                unroll=True,
                            )

                        n_hit, n_cp, n_tot = lax.fori_loop(0, B, row_ecopy, carry, unroll=2)
                    else:
                        for b in range(B):
                            carry = lax.fori_loop(
                                0,
                                c.TOPK,
                                functools.partial(pick_ecopy, b),
                                carry,
                                unroll=True,
                            )
                        n_hit, n_cp, n_tot = carry

            with sc("route_dedup", 2):
                def cp_fields(j):
                    if RSHRINK:
                        return cp_dec(scr["cplist"][j])
                    e = scr["cplist"][j]
                    r0 = lax.div(e, ER)
                    s0 = e - r0 * ER
                    c0 = scr["ecopy"][slot, 0, e].astype(I32) - 1
                    r1 = lax.div(c0, EXTRA)
                    s1 = ER + (c0 - r1 * EXTRA)
                    return e, r0, s0, r1, s1

                def po2c_pass1(j, _):
                    e, r0, _s0, r1, _s1 = cp_fields(j)
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
                    e, r0, s0, r1, s1 = cp_fields(j)
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

        else:
            def pick(b, k, n):
                # branch-free: a repeated (or, for EP, remote) pick rewrites slot n and stamps
                # again without advancing n
                e = scr["hits"][b, k]
                hi, lo = lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)
                seen = scr["seen"][hi, lo]
                new = seen != stamp
                if not c.moe_tp:
                    new = jnp.logical_and(new, lax.div(e, ER) == me)
                if row_mask:  # a masked row's pick neither counts nor stamps
                    on = mr_ref[b] != 0
                    new = jnp.logical_and(new, on)
                    scr["seen"][hi, lo] = jnp.where(on, stamp, seen)
                else:
                    scr["seen"][hi, lo] = stamp
                scr["elist"][n] = e
                return n + jnp.where(new, 1, 0).astype(I32)

            n_hit = jnp.int32(0)
            with sc("route_dedup", 2):
                if WIDE:  # up to 640 picks: a row loop (TOPK picks unrolled per row)
                    def row(b, n):
                        return lax.fori_loop(
                            0, c.TOPK, functools.partial(pick, b), n, unroll=True
                        )

                    n_hit = lax.fori_loop(0, B, row, n_hit, unroll=2)
                else:
                    for b in range(B):
                        n_hit = lax.fori_loop(
                            0, c.TOPK, functools.partial(pick, b), n_hit, unroll=True
                        )
        if "experts" in skip:
            n_hit = jnp.int32(0)
        scr["st"][0] = n_hit * TE
        with sc("route_fetch", 2):
            if VEC and EARLY_FILL:
                self.rings.fetch_range(n_fill0, jnp.minimum(NBE, n_hit * TE), l)
            elif c.stack_e:
                for k in range(NBE):
                    @pl.when(k < n_hit * TE)
                    def _(k=k):
                        fetch_e(k, l)
            else:
                if c.B <= 40:
                    fetch_loop(jnp.minimum(NBE, n_hit * TE), l)
                else:
                    lax.fori_loop(
                        0,
                        jnp.minimum(NBE, n_hit * TE),
                        lambda k, _: (fetch_e(k, l), 0)[1],
                        0,
                    )

        return n_hit

    moe_route = route

    def moe_wide(self, l, slot, seg_r):
        """Wide step MoE over every microbatch's rows: router, shared expert and routed experts
        read scr["xb_all"], each hit expert's weights stream once per layer, and scr["part_all"]
        holds the MoE output of all rows (the all-reduce and residual add follow per microbatch).

        Rows are routed in 128-row chunks (the top-k machinery is built for at most 128 rows);
        the deduplicated hit set of the whole step goes through the usual EP dedup and Po2C
        copy choice. Expert e then gathers its rows with a one-hot matmul into CAP-row tiles
        (groups of CAP rows when it has more), runs the expert on the tile and scatters the
        weighted rows back with the transposed one-hot (the routing weight split into a bf16
        high and low part, so the f32 weight is applied exactly to 2^-16).
        """
        ctx = self.ctx
        c, tiles, scr, sc, sm, mm = ctx.c, ctx.tiles, ctx.scr, ctx.sc, ctx.sm, ctx.mm
        dtake, fetch, fetch_range = self.rings.dtake, self.rings.fetch, self.rings.fetch_range
        etake, egive = self.rings.etake, self.rings.egive
        E, H, IR, ER, R, EXTRA = ctx.E, ctx.H, ctx.IR, ctx.ER, ctx.R, ctx.EXTRA
        L, NBD, NBE, NSS, T13, T2, TE, tk, nt = (
            ctx.L, ctx.NBD, ctx.NBE, ctx.NSS, ctx.T13, ctx.T2, ctx.TE, ctx.tk, ctx.nt)
        BWP, BLIVE, CAP, N_MOE = ctx.BWP, ctx.BLIVE, ctx.CAP, ctx.N_MOE
        EM, FLT_MIN, me, skip = ctx.EM, ctx.FLT_MIN, ctx.me, ctx.skip
        forced_routing, rb_hbm = ctx.forced_routing, ctx.rb_hbm
        MROW = E // 128
        NCP = R * EXTRA
        NQ = BWP // 128
        assert ctx.ECOPY and tiles.route_vec and E % 128 == 0 and MROW + 1 <= ctx.MR, (
            "the wide step uses the vector dedup (EP with copy slots)")
        assert NCP + R + 1 <= 128 and tiles.topk and ctx.topk_lanes and c.TOPK <= 128
        assert N_MOE <= NBD, (N_MOE, NBD)
        xb_all, part_all, wnT = scr["xb_all"], scr["part_all"], scr["wnT_all"]
        if ctx.WIDE_TILE_REUSE:
            xgw, midw, accw = scr["xb"], scr["xb2"], scr["acc"]  # idle microbatch buffers
        else:
            xgw, midw, accw = scr["xgw"], scr["midw"], scr["accw"]
        lane_e = lax.broadcasted_iota(I32, (128, E), 1)
        row_q = lax.broadcasted_iota(I32, (128, 1), 0)
        lane_1e = lax.broadcasted_iota(I32, (1, E), 1)
        lane_bw = lax.broadcasted_iota(I32, (1, BWP), 1)
        row_c = lax.broadcasted_iota(I32, (CAP, 1), 0)
        sub_e = lax.broadcasted_iota(I32, (E, 1), 0)

        g_r = seg_r
        g_13 = seg_r + nt["router"]
        g_2 = g_13 + nt["sh13"]
        with sc("moe_router"):
            w_r = [dtake("router", g_r + t) for t in range(nt["router"])]
            w_13 = [dtake("sh13", g_13 + t) for t in range(nt["sh13"])]
            w_2 = [dtake("sh2", g_2 + t) for t in range(nt["sh2"])]
        tk_r, tk_13, tk_2 = tk["router"], tk["sh13"], tk["sh2"]
        sh_gate, sh13_s, sh2_s = sm(slot, "sh_gate"), sm(slot, "sh13_s"), sm(slot, "sh2_s")
        hitf = jnp.zeros((1, E), I32)
        cnt = jnp.zeros((1, E), I32)

        def rb_copy(l_, q_):
            """Routing-bias chunk q_ of layer l_ into slot (l_ * NQ + q_) % 2."""
            rs = lax.rem(l_ * NQ + q_, 2)
            return pltpu.make_async_copy(
                rb_hbm.at[l_, pl.ds(q_ * 128, 128)], scr["rb"].at[rs], scr["rb_sem"].at[rs]
            )

        for q in range(NQ):
            r0 = q * 128
            with sc("moe_router"):
                x = xb_all[r0:r0 + 128, :]
                logits = None
                for t in range(nt["router"]):
                    p_ = mm(x[:, t * tk_r:(t + 1) * tk_r], w_r[t][...])
                    logits = p_ if logits is None else logits + p_
                xf = x.astype(F32)
                glog = jnp.sum(xf * sh_gate, axis=-1, keepdims=True)
                sg = jax.nn.sigmoid(glog)
                mx = jnp.max(logits, axis=-1, keepdims=True)
                pexp = jnp.exp(logits - mx)
            with sc("moe_shared"):
                g13 = None
                for t in range(nt["sh13"]):
                    p_ = mm(x[:, t * tk_13:(t + 1) * tk_13], w_13[t][...])
                    g13 = p_ if g13 is None else g13 + p_
                g13 = g13 * sh13_s
                mid = rnd(rnd(jax.nn.silu(rnd(g13[:, :c.ISR]))) * rnd(g13[:, c.ISR:])).astype(BF16)
                ysh = None
                for t in range(nt["sh2"]):
                    p_ = mm(mid[:, t * tk_2:(t + 1) * tk_2], w_2[t][...])
                    ysh = p_ if ysh is None else ysh + p_
                part_all[r0:r0 + 128, :] = ysh * sh2_s * sg
            with sc("moe_route"):
                choice = pexp
                if forced_routing:
                    rb_copy(l, q).wait()
                    choice = scr["rb"][lax.rem(l * NQ + q, 2)]
                    if q + 1 < NQ:
                        rb_copy(l, q + 1).start()
                    else:
                        pl.when(l + 1 < L)(lambda: rb_copy(l + 1, 0).start())
                live = row_q < (BLIVE - r0)
                kb = lax.bitcast_convert_type(jnp.maximum(choice, FLT_MIN), I32)
                key = lax.bitcast_convert_type((kb & ~EM) | (EM - lane_e), F32)
                _, sel = topk_rows_on_lanes(key, c.TOPK, EM, need_ids=False)
                sel = jnp.logical_and(sel, live)
                wsel = jnp.where(sel, pexp, 0.0)
                wn = jnp.where(live, wsel / jnp.sum(wsel, axis=-1, keepdims=True), 0.0)
                wnT[:, r0:r0 + 128] = wn.T
                seli = sel.astype(I32)
                hitf = jnp.maximum(hitf, jnp.max(seli, axis=0, keepdims=True))
                cnt = cnt + jnp.sum(seli, axis=0, keepdims=True)
        if "experts" in skip:
            hitf = jnp.zeros_like(hitf)
        # the MoE dense tiles are consumed: refill their slots (lazy)
        with sc("refill"):
            for t in range(N_MOE):
                fetch(seg_r + t + NBD)

        # ---- dedup and Po2C over the step's hit set (the vector route of `route`) ----
        with sc("route_topk", 2):
            cpi = sm(slot, "ecopy").astype(I32)  # [1, E]: 0 or 1 + copy slot
            fixed = jnp.where(cpi > 0, 0, hitf)  # hit, no redundant copy
            for j in range(MROW):
                scr["hitv"][j:j + 1, :] = fixed[:, j * 128:(j + 1) * 128]
            v1 = (lane_1e + 1) * hitf
            row_a = lax.broadcasted_iota(I32, (16, E), 0)
            a = jnp.where(
                row_a == 0,
                lax.shift_right_logical(v1, 5),
                jnp.where(row_a == 1, lax.bitwise_and(v1, 31), jnp.where(row_a == 2, hitf, 0)),
            )
            a = a.astype(F32).astype(BF16)
            row_b = lax.broadcasted_iota(I32, (128, E), 0)
            lane_b = lax.broadcasted_iota(I32, (128, E), 1)
            rr = row_b - NCP
            in_rank = jnp.logical_and(lane_b >= rr * ER, lane_b < (rr + 1) * ER)
            cp_code = (row_b % R) * EXTRA + lax.div(row_b, R) + 1
            bm = jnp.where(
                row_b < NCP,
                jnp.where(cpi == cp_code, 1.0, 0.0),
                jnp.where(rr < R, jnp.where(in_rank, 1.0, 0.0), jnp.where(rr == R, 1.0, 0.0)),
            ).astype(BF16)
            res = mm(a, bm, NT)  # [16, 128]
            lane_r = lax.broadcasted_iota(I32, (1, 128), 1)
            meta = jnp.where(lane_r < NCP, res[0:1] * 32.0 + res[1:2], res[2:3])
            scr["hitv"][MROW:MROW + 1, :] = meta.astype(I32)
            for j in range(MROW):  # rows per expert, read by the expert loop from hits
                scr["hitv"][MROW + 1 + j:MROW + 2 + j, :] = cnt[:, j * 128:(j + 1) * 128]
        with sc("route_dma", 2):
            cp = pltpu.make_async_copy(scr["hitv"], scr["hits"], scr["misc_sem"])
            cp.start()
            cp.wait()
        RSHRINK = bool(ctx.ROUTE_SHRINK and E < (1 << ctx.CP_SH))
        PRIM_NOW_K = ctx.PRIM_NOW_K
        CP_SH = ctx.CP_SH

        def cp_dec(w):
            e = lax.bitwise_and(w, (1 << CP_SH) - 1)
            j = lax.shift_right_logical(w, CP_SH)
            if ER & (ER - 1) == 0:
                r0_ = lax.shift_right_logical(e, ER.bit_length() - 1)
                s0 = lax.bitwise_and(e, ER - 1)
            else:
                r0_ = lax.div(e, ER)
                s0 = e - r0_ * ER
            if R & (R - 1) == 0:
                r1 = lax.bitwise_and(j, R - 1)
                s1 = ER + lax.shift_right_logical(j, R.bit_length() - 1)
            else:
                r1 = lax.rem(j, R)
                s1 = ER + lax.div(j, R)
            return e, r0_, s0, r1, s1

        with sc("route_dedup", 2):
            for r in range(R):
                scr["rload"][r] = scr["hits"][MROW, NCP + r]
            n_tot = scr["hits"][MROW, NCP + R]
            base = me * ER

            def prim(s_, n):
                e = base + s_
                f = scr["hits"][lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)]
                scr["elist"][n] = e
                if not PRIM_NOW_K:
                    scr["wlist"][n] = s_
                return n + f

            n_hit = lax.fori_loop(0, ER, prim, jnp.int32(0), unroll=True)
            if PRIM_NOW_K:
                scr["st"][5] = n_hit
            EARLY_FILL = "experts" not in skip
            n_fill0 = jnp.int32(0)
            if EARLY_FILL:
                n_fill0 = jnp.minimum(NBE, n_hit * TE)
                with sc("route_fetch0", 2):
                    fetch_range(jnp.int32(0), n_fill0, l, fixed=PRIM_NOW_K)
            if RSHRINK:
                n_cp = jnp.int32(0)
                for c0 in range(0, NCP, 16):
                    vs = [scr["hits"][MROW, j] for j in range(c0, min(c0 + 16, NCP))]
                    for k, v1_ in enumerate(vs):
                        scr["cplist"][n_cp] = v1_ + (((c0 + k) << CP_SH) - 1)
                        n_cp = n_cp + lax.shift_right_logical(-v1_, 31)
            else:
                def cpl(j, n):
                    v1_ = scr["hits"][MROW, j]
                    scr["cplist"][n] = v1_ - 1
                    return n + jnp.where(v1_ > 0, 1, 0).astype(I32)

                n_cp = lax.fori_loop(0, NCP, cpl, jnp.int32(0), unroll=True)

            def cp_fields(j):
                if RSHRINK:
                    return cp_dec(scr["cplist"][j])
                e = scr["cplist"][j]
                r0_ = lax.div(e, ER)
                s0 = e - r0_ * ER
                c0 = scr["ecopy"][slot, 0, e].astype(I32) - 1
                r1 = lax.div(c0, EXTRA)
                s1 = ER + (c0 - r1 * EXTRA)
                return e, r0_, s0, r1, s1

            def po2c_pass1(j, _):
                e, r0_, _s0, r1, _s1 = cp_fields(j)
                l0 = scr["rload"][r0_]
                l1 = scr["rload"][r1]
                use_c = jnp.logical_or(
                    l1 + 1 < l0,
                    jnp.logical_and(l1 < l0, jnp.logical_or(l0 * R > n_tot, l1 * R < n_tot)),
                )
                d = jnp.where(use_c, 1, 0).astype(I32)
                scr["rload"][r0_] = l0 - d
                scr["rload"][r1] = l1 + d
                scr["cpchoice"][j] = d
                return 0

            def po2c_pass2(j, n):
                e, r0_, s0, r1, s1 = cp_fields(j)
                use_c = scr["cpchoice"][j] != 0
                cur = jnp.where(use_c, r1, r0_)
                alt = jnp.where(use_c, r0_, r1)
                l_cur = scr["rload"][cur]
                l_alt = scr["rload"][alt]
                flip = jnp.logical_or(
                    l_alt + 1 < l_cur,
                    jnp.logical_and(
                        l_alt < l_cur, jnp.logical_and(l_cur * R > n_tot, l_alt * R < n_tot)
                    ),
                )
                d = jnp.where(flip, 1, 0).astype(I32)
                scr["rload"][cur] = l_cur - d
                scr["rload"][alt] = l_alt + d
                final_c = jnp.not_equal(use_c, flip)
                pick_r = jnp.where(final_c, r1, r0_)
                pick_w = jnp.where(final_c, s1, s0)
                mine = pick_r == me
                scr["elist"][n] = e
                scr["wlist"][n] = pick_w
                return n + jnp.where(mine, 1, 0).astype(I32)

            lax.fori_loop(0, n_cp, po2c_pass1, 0)
            n_hit = lax.fori_loop(0, n_cp, po2c_pass2, n_hit)
        if "experts" in skip:
            n_hit = jnp.int32(0)
        scr["st"][0] = n_hit * TE
        with sc("route_fetch", 2):
            if EARLY_FILL:
                fetch_range(n_fill0, jnp.minimum(NBE, n_hit * TE), l)

        # ---- expert-major loop: each hit expert once over the rows of every microbatch ----
        TK13, TK2 = tk["e13"], tk["e2"]
        BN = 2 * IR
        sh_hi = [1 << k for k in range(7)]

        def lane_scan(v):
            """Inclusive prefix sum of v [1, 128] (i32) along lanes."""
            lane = lax.broadcasted_iota(I32, (1, 128), 1)
            for sh in sh_hi:
                v = v + jnp.where(lane >= sh, pltpu.roll(v, sh, 1), 0)
            return v

        def expert_w(i, carry):
            e = scr["elist"][i]
            s_ = i % NSS
            base_u = i * TE
            # the expert's rows: its routing-weight row over every row of the step
            if ctx.WIDE_DYN_ROW:
                col = wnT[pl.ds(e, 1), :]
            else:
                col = jnp.sum(jnp.where(sub_e == e, wnT[...], 0.0), axis=0, keepdims=True)
            nz = col != 0.0
            nzi = nz.astype(I32)
            # exclusive prefix count of the hit rows, 128 lanes at a time
            segs = []
            carry_n = jnp.zeros((1, 1), I32)
            for q in range(NQ):
                inc = lane_scan(nzi[:, q * 128:(q + 1) * 128])
                segs.append(inc + carry_n)
                carry_n = carry_n + inc[:, 127:128]
            pos = jnp.concatenate(segs, axis=1) - nzi if NQ > 1 else segs[0] - nzi
            n_e = scr["hits"][MROW + 1 + lax.shift_right_logical(e, 7), lax.bitwise_and(e, 127)]
            n_grp = lax.div(n_e + (CAP - 1), CAP)
            w13 = [etake("e13", base_u + t) for t in range(T13)]
            self.es_wait(i)
            es = scr["es"][s_]  # [1, 2 * IR + H]
            w2 = [etake("e2", base_u + T13 + t) for t in range(T2)]

            def group(g, carry_g):
                last = g == n_grp - 1
                with sc("moe_gather"):
                    sel_g = jnp.logical_and(pos - g * CAP == row_c, nz)  # [CAP, BWP]
                    gm = sel_g.astype(BF16)
                    w = jnp.sum(jnp.where(sel_g, col, 0.0), axis=1, keepdims=True)  # [CAP, 1]
                    # one-hot gather in 128-row pieces of the contraction (small temporaries)
                    xg = None
                    for q in range(NQ):
                        p_ = mm(gm[:, q * 128:(q + 1) * 128], xb_all[q * 128:(q + 1) * 128, :])
                        xg = p_ if xg is None else xg + p_
                    xgw[0:CAP, :] = xg.astype(BF16)
                with sc("moe_experts"):
                    for t in range(T13):
                        m13 = mm(xgw[0:CAP, t * TK13:(t + 1) * TK13], w13[t][...])
                        if t == 0:
                            accw[0:CAP, 0:BN] = m13
                        else:
                            accw[0:CAP, 0:BN] += m13
                        pl.when(last)(lambda t=t: egive(base_u + t, l, t))
                    h = accw[0:CAP, 0:BN] * es[:, :BN]
                    midw[0:CAP, 0:IR] = rnd(
                        rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])
                    ).astype(BF16)
                    for t in range(T2):
                        m2 = mm(midw[0:CAP, t * TK2:(t + 1) * TK2], w2[t][...])
                        if t == 0:
                            accw[0:CAP, 0:H] = m2
                        else:
                            accw[0:CAP, 0:H] += m2
                        pl.when(last)(lambda t=t: egive(base_u + T13 + t, l, T13 + t))
                    y = rnd(accw[0:CAP, 0:H] * es[:, BN:])  # [CAP, H], bf16-exact
                with sc("moe_scatter"):
                    w_hi = w.astype(BF16).astype(F32)
                    w_lo = w - w_hi
                    gmf = sel_g.astype(F32)
                    gw = jnp.concatenate([gmf * w_hi, gmf * w_lo], axis=0)  # [2 * CAP, BWP] f32
                    y2 = jnp.concatenate([y, y], axis=0).astype(BF16)  # [2 * CAP, H]
                    # transposed one-hot scatter-add in 64-row pieces of part_all
                    if not ctx.WIDE_SCATTER:
                        part_all[0:8, 0:128] += 0.0 * y[0:8, 0:128]  # keeps the expert work live
                    elif ctx.WIDE_SCATTER_TN:
                        gwb = gw.astype(BF16)
                        for r0 in range(0, BWP, 64):
                            part_all[r0:r0 + 64, :] += lax.dot_general(
                                gwb[:, r0:r0 + 64], y2, (((0,), (0,)), ((), ())),
                                preferred_element_type=F32,
                            )
                    else:
                        gwT = gw.T.astype(BF16)  # [BWP, 2 * CAP]
                        for r0 in range(0, BWP, 64):
                            part_all[r0:r0 + 64, :] += mm(gwT[r0:r0 + 64, :], y2)
                return carry_g

            lax.fori_loop(0, n_grp, group, 0)
            return carry

        with sc("moe_experts"):
            lax.fori_loop(0, n_hit, expert_w, 0)

    def expert(self, l, i, carry):
        """Listed expert i on all rows: part += wn[:, e] * expert_e(xb). EP: the whole expert;
        TP: this rank's slice of the intermediate dim (the all-reduce sums the slices)."""
        ctx = self.ctx
        c, scr, mm = ctx.c, ctx.scr, ctx.mm
        etake, egive, egive_group = (
            self.rings.etake,
            self.rings.egive,
            self.rings.egive_group,
        )
        H, IR, CH, MR = ctx.H, ctx.IR, ctx.CH, ctx.MR
        NSS, T13, T2, TE, tk = ctx.NSS, ctx.T13, ctx.T2, ctx.TE, ctx.tk
        lane_e8 = ctx.lane_e8

        e = scr["elist"][i]
        s = i % NSS
        base = i * TE
        acc, xb, xb2, part = scr["acc"], scr["xb"], scr["xb2"], scr["part"]
        if T13 == 1:
            h = mm(xb[...], etake("e13", base)[...])
            egive(base, l)
        else:
            # Hypothesis expert-mrb: static unroll (like the PIPE_EP prologue) so the MXU
            # accumulates the T13 partial products in the MRB and drains once per projection
            # instead of once per tile. Same summation order: ((m0 + m1) + m2) + ...
            TK13 = tk["e13"]
            for t in range(T13):
                u = base + t
                w = etake("e13", u)
                m13 = mm(xb[:, t * TK13:(t + 1) * TK13], w[...])
                if t == 0:
                    acc[:, :2 * IR] = m13
                else:
                    acc[:, :2 * IR] += m13
                if c.B <= 40:
                    if t % 2 == 1:
                        egive_group([(u - 1, t - 1), (u, t)], l)
                    elif t == T13 - 1:
                        egive(u, l, t)
                else:
                    egive(u, l, t)
            h = acc[:, :2 * IR]
        self.es_wait(i)
        h = h * scr["es"][s, :, :2 * IR]
        mid_ = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:]))
        xb2[:, :IR] = mid_.astype(BF16)
        wcol = jnp.sum(
            jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True
        )
        u2 = base + T13
        if T2 == 1:
            # one [IR, H] tile: CH-lane chunks keep each result in a few vregs
            w2 = etake("e2", u2)[...]
            xm = xb2[:, :IR]
            for j in range(H // CH):
                lo = 2 * IR + j * CH
                y = mm(xm, w2[:, j * CH:(j + 1) * CH])[0:MR]
                y = rnd(y * scr["es"][s, :, lo:lo + CH])
                part[0:MR, j * CH:(j + 1) * CH] += wcol * y
            egive(u2, l)
        else:
            TK2 = tk["e2"]
            for t in range(T2):
                u = u2 + t
                w = etake("e2", u)
                m2 = mm(xb2[:, t * TK2:(t + 1) * TK2], w[...])
                if t == 0:
                    acc[:, :H] = m2
                else:
                    acc[:, :H] += m2
                if c.B <= 40:
                    if t % 2 == 1:
                        egive_group([(u - 1, T13 + t - 1), (u, T13 + t)], l)
                    elif t == T2 - 1:
                        egive(u, l, T13 + t)
                else:
                    egive(u, l, T13 + t)
            y = rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR:])
            part[0:MR, :] += wcol * y
        return carry

    def gate_up(self, i, w13):
        """Single-tile expert i's activated gate|up rows [MB, IR] (bf16)."""
        ctx = self.ctx
        scr, mm, NSS, IR = ctx.scr, ctx.mm, ctx.NSS, ctx.IR
        h = mm(scr["xb"][...], w13[...]) * scr["es"][i % NSS, :, :2 * IR]
        return rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)

    def down(self, i, mid, w2):
        """part[0:MR] += wn[:, e] * (mid @ e2) for single-tile expert i, in CH-lane chunks."""
        ctx = self.ctx
        scr, mm, NSS, IR, H, CH, MR, lane_e8 = (
            ctx.scr,
            ctx.mm,
            ctx.NSS,
            ctx.IR,
            ctx.H,
            ctx.CH,
            ctx.MR,
            ctx.lane_e8,
        )
        e, s, part = scr["elist"][i], i % NSS, scr["part"]
        wcol = jnp.sum(
            jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True
        )
        for j in range(H // CH):
            lo = 2 * IR + j * CH
            y = mm(mid, w2[:, j * CH:(j + 1) * CH])[0:MR]
            y = rnd(y * scr["es"][s, :, lo:lo + CH])
            part[0:MR, j * CH:(j + 1) * CH] += wcol * y

    def experts_piped(self, l, n_hit):
        """Single-tile experts, software-pipelined: iteration i runs expert i+1's gate|up
        next to expert i's down projection (independent MXU chains). Ring waits come first
        and refills last, so neither splits the compute."""
        ctx = self.ctx
        scr, IR, T13, TE = ctx.scr, ctx.IR, ctx.T13, ctx.TE
        etake, egive = self.rings.etake, self.rings.egive
        xb2 = scr["xb2"]

        @pl.when(n_hit > 0)
        def _():
            w13 = etake("e13", 0)
            self.es_wait(0)
            xb2[:, :IR] = self.gate_up(0, w13)
            egive(0, l, 0)

            def body(i, carry):
                u13, u2 = (i + 1) * TE, i * TE + T13
                w13 = etake("e13", u13)
                self.es_wait(i + 1)
                w2 = etake("e2", u2)[...]
                mid = xb2[:, :IR]
                nxt = self.gate_up(i + 1, w13)
                self.down(i, mid, w2)
                xb2[:, :IR] = nxt
                egive(u13, l, 0)
                egive(u2, l, T13)
                return carry

            lax.fori_loop(0, n_hit - 1, body, 0)
            last = n_hit - 1
            u2 = last * TE + T13
            self.down(last, xb2[:, :IR], etake("e2", u2)[...])
            egive(u2, l, T13)

    def experts_piped_ep(self, l, n_hit, pre_drain=None):
        """EP multi-tile software pipeline (T13 == 2 * T2, e.g. T13=8, T2=4, TE=12, NBE=12):
        Prologue computes expert 0's gate|up into xb2[:, :IR]. Each steady-state step i overlaps
        expert i's 4 w2 tiles with expert (i+1)'s 8 w13 tiles (1 w2 tile + 2 w13 tiles per
        macro-step s_step in 0..T2-1) using scr['acc13'] for expert i+1's gate|up accumulator.
        When tiles.r2_direct_store is True, s_step == 0 writes directly (=) instead of zero-filling
        acc13 and acc beforehand and accumulating (+=), saving 2 full VMEM zero-writes and 2 VMEM
        RMWs per active expert."""
        ctx = self.ctx
        c, tiles, scr, mm = ctx.c, ctx.tiles, ctx.scr, ctx.mm
        etake, egive, egive_group = (
            self.rings.etake,
            self.rings.egive,
            self.rings.egive_group,
        )
        H, IR, MB, MR = ctx.H, ctx.IR, ctx.MB, ctx.MR
        NSS, T13, T2, TE, tk = ctx.NSS, ctx.T13, ctx.T2, ctx.TE, ctx.tk
        lane_e8 = ctx.lane_e8

        @pl.when(n_hit > 0)
        def _():
            acc, acc13 = scr["acc"], scr["acc13"]
            xb, xb2, part = scr["xb"], scr["xb2"], scr["part"]
            BN = 2 * IR
            TK13, TK2 = tk["e13"], tk["e2"]
            DIRECT0 = tiles.r2_direct_store

            # Prologue: expert 0 gate|up (T13 tiles)
            if not DIRECT0:
                acc13[...] = jnp.zeros((MB, BN), F32)
            for t in range(T13):
                w13 = etake("e13", t)
                k0 = t * TK13
                mm13 = mm(xb[:, k0 : k0 + TK13], w13[...])
                if DIRECT0 and t == 0:
                    acc13[:, :BN] = mm13
                else:
                    acc13[:, :BN] += mm13
                if c.B <= 40:
                    if t % 2 == 1:
                        egive_group([(t - 1, t - 1), (t, t)], l)
                    elif t == T13 - 1:
                        egive(t, l, t)
                else:
                    egive(t, l, t)
            self.es_wait(0)
            h0 = acc13[:, :BN] * scr["es"][0, :, :BN]
            mid0 = rnd(rnd(jax.nn.silu(rnd(h0[:, :IR]))) * rnd(h0[:, IR:]))
            xb2[:, :IR] = mid0.astype(BF16)

            # EP_FAST_REFILL: with one expert per ring (NBE == TE) unit u = i * TE + t sits in
            # the static bank t, so the steady-state refills of iteration i are expert i + 1's
            # w2 tiles and expert i + 2's w13 tiles into known banks: two slot lookups and two
            # bounds checks per expert and ONE layer-dispatch walk per macro-step, instead of a
            # per-tile unit decode (div / rem / SMEM list loads, bounds check) and walk. Same
            # DMAs into the same banks, issued after the macro-step's 3 tile matmuls.
            FAST = bool(
                getattr(_cfg, "EP_FAST_REFILL", False) and ctx.NBE == TE and not c.stack_e
            )

            def body_ep(i, _):
                e = scr["elist"][i]
                s_cur = i % NSS
                s_nxt = (i + 1) % NSS
                base_w2 = i * TE + T13
                base_nxt13 = (i + 1) * TE
                if FAST:
                    w_n1 = self.rings.slot_of(i + 1)
                    w_n2 = self.rings.slot_of(i + 2)
                    g_n1 = i + 1 < n_hit
                    g_n2 = i + 2 < n_hit

                if not DIRECT0:
                    acc[:, :H] = jnp.zeros((MB, H), F32)
                    acc13[...] = jnp.zeros((MB, BN), F32)

                for s_step in range(T2):
                    u2 = base_w2 + s_step
                    t0 = 2 * s_step
                    t1 = 2 * s_step + 1
                    u13_0 = base_nxt13 + t0
                    u13_1 = base_nxt13 + t1

                    w2 = etake("e2", u2)
                    w13_0 = etake("e13", u13_0)
                    k2 = s_step * TK2
                    k13_0 = t0 * TK13
                    mm2 = mm(xb2[:, k2 : k2 + TK2], w2[...])
                    mm13_0 = mm(xb[:, k13_0 : k13_0 + TK13], w13_0[...])
                    if DIRECT0 and s_step == 0:
                        acc[:, :H] = mm2
                        acc13[:, :BN] = mm13_0
                    else:
                        acc[:, :H] += mm2
                        acc13[:, :BN] += mm13_0
                    if FAST:
                        pass
                    elif c.B <= 40:
                        egive_group([(u2, T13 + s_step), (u13_0, t0)], l)
                    else:
                        egive(u2, l, T13 + s_step)
                        egive(u13_0, l, t0)

                    w13_1 = etake("e13", u13_1)
                    k13_1 = t1 * TK13
                    acc13[:, :BN] += mm(xb[:, k13_1 : k13_1 + TK13], w13_1[...])
                    if FAST:
                        self.rings.refill_static(
                            l,
                            [
                                (w_n1, T13 + s_step, T13 + s_step, None),
                                (w_n2, t0, t0, g_n2),
                                (w_n2, t1, t1, g_n2),
                            ],
                            g_n1,
                        )
                        if t0 == 0:
                            pl.when(g_n2)(
                                lambda: self.rings._es_start(i + 2, w_n2, l)
                            )
                    else:
                        egive(u13_1, l, t1)

                wcol = jnp.sum(
                    jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True
                )
                y = rnd(acc[0:MR, :H] * scr["es"][s_cur, :, BN:])
                part[0:MR, :] += wcol * y

                self.es_wait(i + 1)
                h_nxt = acc13[:, :BN] * scr["es"][s_nxt, :, :BN]
                mid_nxt = rnd(rnd(jax.nn.silu(rnd(h_nxt[:, :IR]))) * rnd(h_nxt[:, IR:]))
                xb2[:, :IR] = mid_nxt.astype(BF16)
                return 0

            lax.fori_loop(0, n_hit - 1, body_ep, 0)

            # Epilogue: last expert (n_hit - 1) down projection (T2 tiles)
            if pre_drain is not None:
                pre_drain()
            i_last = n_hit - 1
            e_last = scr["elist"][i_last]
            s_last = i_last % NSS
            base_last_w2 = i_last * TE + T13
            if not DIRECT0:
                acc[:, :H] = jnp.zeros((MB, H), F32)
            for s_step in range(T2):
                u2 = base_last_w2 + s_step
                w2 = etake("e2", u2)
                k2 = s_step * TK2
                mm2 = mm(xb2[:, k2 : k2 + TK2], w2[...])
                if DIRECT0 and s_step == 0:
                    acc[:, :H] = mm2
                else:
                    acc[:, :H] += mm2
                if c.B <= 40:
                    if s_step % 2 == 1:
                        egive_group([(u2 - 1, T13 + s_step - 1), (u2, T13 + s_step)], l)
                    elif s_step == T2 - 1:
                        egive(u2, l, T13 + s_step)
                else:
                    egive(u2, l, T13 + s_step)
            wcol_last = jnp.sum(
                jnp.where(lane_e8 == e_last, scr["wn"][...], 0.0), axis=1, keepdims=True
            )
            y_last = rnd(acc[0:MR, :H] * scr["es"][s_last, :, BN:])
            part[0:MR, :] += wcol_last * y_last
