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
"""Qwen3.5-397B-A17B-FP8 whole-backbone decode megakernel for TPU v7x-8.

One `pl.pallas_call` per decode step under `shard_map(mesh=tp=8)`: 60 layers (45 GatedDeltaNet +
15 full attention, 3:1) x (attention + 512-expert top-10 + shared-expert MoE) + final RMSNorm,
2 AllReduces / layer via `pltpu.make_async_remote_copy` over ICI inside the kernel.
"""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .attention_block import PagedAttentionBlock
from .collectives import PallasAllReduce
from .config import (
    ABLATIONS,
    BF16,
    COLLECTIVE_ID,
    CONV_STAGE_MIN_B,
    DEFAULT_TILES,
    DENSE,
    DMM_UNROLL,
    EXPERT,
    F32,
    FP8,
    FULL,
    FULL_EP,
    GQ_PACK,
    I32,
    MKConfig,
    NN,
    NT,
    O_NAMES,
    S_NAMES,
    Spec,
    TINY,
    TINY_EP,
    TINY_TILES,
    TOPK_LANES,
    TOPK_NL,
    Tiles,
    U8,
    U32,
    VERIFY_TILES,
    W_NAMES,
    WIDE_TILES,
    accept_states,
    bswitch,
    default_tiles,
    encode_ecopy,
    fold,
    kernel_out_shapes,
    plan,
    pool_alias,
    rnd,
    rup,
    smalls_layout,
    state_shapes,
    topk_rows_on_lanes,
    unfold,
    weight_shapes,
)
from .context import MKBuildContext
from .gdn_block import GDNBlock
from .moe_block import MoEBlock
from .pool_dma import GDNPoolDMA, GprRecIn, KVPoolDMA
from .rings import DenseWeightRing, ExpertWeightRing, WeightRings


def build(
    c: MKConfig,
    tiles: Tiles | None = None,
    *,
    interpret=False,
    forced_routing=False,
    skip=frozenset(),
    scopes=False,
    row_mask=False,
    pool_gather: bool | None = None,
):
    """Build the single-step `pl.pallas_call` (per-rank body for `shard_map`) and its plan info."""
    ctx = MKBuildContext(
        c,
        tiles,
        interpret=interpret,
        forced_routing=forced_routing,
        skip=skip,
        scopes=scopes,
        row_mask=row_mask,
        pool_gather=pool_gather,
        gq_pack=GQ_PACK,
        topk_lanes=TOPK_LANES,
        dmm_unroll=DMM_UNROLL,
        conv_stage_min_b=CONV_STAGE_MIN_B,
    )
    tiles = ctx.tiles
    R = ctx.R

    def kernel(*refs):
        if ctx.WIDE_MB:
            kernel_wide(ctx, refs)
            return
        ctx.bind_refs(refs)
        rings = WeightRings(ctx)
        allreduce_blk = PallasAllReduce(ctx)
        gdn_dma = GDNPoolDMA(ctx)
        kv_dma = KVPoolDMA(ctx)
        gdn_blk = GDNBlock(ctx, rings, gdn_dma)
        att_blk = PagedAttentionBlock(ctx, rings, kv_dma)
        moe_blk = MoEBlock(ctx, rings)

        scr = ctx.scr
        W, O, hid_ref, out_ref = ctx.W, ctx.O, ctx.hid_ref, ctx.out_ref
        sc, sm, rms_zc = ctx.sc, ctx.sm, ctx.rms_zc
        B, M, H, L, E, AE = ctx.B, ctx.M, ctx.H, ctx.L, ctx.E, ctx.AE
        MB, MR, NS, NBD, NGI, NGO = ctx.MB, ctx.MR, ctx.NS, ctx.NBD, ctx.NGI, ctx.NGO
        FUSED, GCH, ECOPY, SPLIT, GQ, PACK, QV = (
            ctx.FUSED,
            ctx.GCH,
            ctx.ECOPY,
            ctx.SPLIT,
            ctx.GQ,
            ctx.PACK,
            ctx.QV,
        )
        t_block, t_gdn, lay = ctx.t_block, ctx.t_gdn, ctx.lay
        me, live, rb_hbm = ctx.me, ctx.live, ctx.rb_hbm

        # ------------------------------ layer loop ------------------------------
        def layer(l, carry):
            slot = l % 2
            with sc("norm_in"):
                pltpu.make_async_copy(
                    W["smalls"].at[0], scr["sm"].at[slot], scr["sm_sem"].at[slot]
                ).wait()
                if ECOPY:
                    o_ec = lay["ecopy"][0]
                    pltpu.make_async_copy(
                        W["smalls"].at[0, :, pl.ds(o_ec, E)],
                        scr["ecopy"].at[slot],
                        scr["ec_sem"].at[slot],
                    ).wait()

                @pl.when(l + 1 < L)
                def _():
                    pltpu.make_async_copy(
                        W["smalls"].at[l + 1],
                        scr["sm"].at[1 - slot],
                        scr["sm_sem"].at[1 - slot],
                    ).start()
                    if ECOPY:
                        o_ec = lay["ecopy"][0]
                        pltpu.make_async_copy(
                            W["smalls"].at[l + 1, :, pl.ds(o_ec, E)],
                            scr["ecopy"].at[1 - slot],
                            scr["ec_sem"].at[1 - slot],
                        ).start()

                is_att = l % AE == AE - 1
                g0 = (l // AE) * t_block + (l % AE) * t_gdn
                if ctx.ARP:  # own half now (its residual was added by the previous MoE AR)
                    allreduce_blk.arp_norm_half(slot, ctx.arp_c, True)
                else:
                    scr["xb"][...] = rms_zc(
                        scr["res"][...], sm(slot, "in_norm")
                    ).astype(BF16)

            first = l == 0
            hook = None
            if ctx.ARP:
                def hook():  # inside the input projection, after the own-half K tiles
                    allreduce_blk.arp_round2_add()  # layer 0: the primed zero round 2
                    allreduce_blk.arp_norm_half(slot, 1 - ctx.arp_c, False)

            def att_branch():
                if ctx.AR0S and ctx.comm:
                    att_blk.attention(
                        l, slot, g0, hook=hook, ar0=allreduce_blk.ar0_spec(slot, first)
                    )
                    allreduce_blk.ar0_finish(0, first)
                    return
                allreduce_blk.add_residual(
                    allreduce_blk.allreduce(
                        0, att_blk.attention(l, slot, g0, hook=hook), first
                    )
                )

            def gdn_branch():
                if ctx.AR0S and ctx.comm:
                    gdn_blk.gdn(
                        l, slot, g0, hook=hook, ar0=allreduce_blk.ar0_spec(slot, first)
                    )
                    allreduce_blk.ar0_finish(0, first)
                    return
                allreduce_blk.add_residual(
                    allreduce_blk.allreduce(
                        0, gdn_blk.gdn(l, slot, g0, hook=hook), first
                    )
                )

            if AE == 1:
                att_branch()
            else:
                lax.cond(is_att, att_branch, gdn_branch)
            y_moe, fill = moe_blk.moe(l, slot, g0, is_att, defer_flush=True)
            if ctx.ARP:  # own half's residual added; round 2 waited inside the next layer's mixer
                allreduce_blk.allreduce(1, y_moe, first, fill=fill, defer=True)
            else:
                allreduce_blk.add_residual(
                    allreduce_blk.allreduce(1, y_moe, first, fill=fill)
                )
            return carry

        # ------------------------------ step ------------------------------
        scr["hitv"][...] = jnp.zeros((MR, 128), I32)
        zero_seen = pltpu.make_async_copy(
            scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"]
        )
        zero_seen.start()
        for i in range(NBD):
            rings.fetch(jnp.int32(i))
        pltpu.make_async_copy(
            W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
        ).start()
        if ECOPY:
            o_ec = lay["ecopy"][0]
            pltpu.make_async_copy(
                W["smalls"].at[0, :, pl.ds(o_ec, E)],
                scr["ecopy"].at[0],
                scr["ec_sem"].at[0],
            ).start()
        if ctx.forced_routing:
            pltpu.make_async_copy(
                rb_hbm.at[0], scr["rb"].at[0], scr["rb_sem"].at[0]
            ).start()
        att_blk.init_worklist()
        if R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, R):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, R - 1)
        scr["res"][...] = jnp.where(live, hid_ref[:MB].astype(F32), 0.0)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        if c.paged_kv:
            scr["knew_o"][...] = jnp.zeros((MR, ctx.HD), FP8)
            scr["vnew_o"][...] = jnp.zeros((MR, ctx.HD), FP8)
        if FUSED:
            for s in range(NS):
                scr["kv_rd_pend"][s] = jnp.int32(0)
                scr["kv_wr_pend"][s] = jnp.int32(0)
            if c.NGDN > 0:
                for slot_i in range(NGI):
                    scr["gdn_in_pend"][slot_i] = jnp.int32(0)
                for slot_o in range(NGO):
                    scr["gdn_out_mode"][slot_o] = jnp.int32(0)
                scr["gdn_out_page"][...] = jnp.zeros(scr["gdn_out_page"].shape, FP8)
        if SPLIT and not GQ:
            scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)
        if PACK or QV:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
        att_blk.init_rope()
        if ctx.ARP:
            # ARPIPE prime: a zero round 2 of the MoE all-reduce, so layer 0's input projection
            if ctx.HIER == 1:  # zero push half 1 from every chip, own entry included
                scr["ag_send"][1] = jnp.zeros(scr["ag_send"].shape[1:], BF16)
                scr["ag_recv"][1, me // 2] = jnp.zeros(scr["ag_recv"].shape[2:], BF16)
            else:
                scr["s2_send"][1] = jnp.zeros(scr["s2_send"].shape[1:], BF16)
            for d in allreduce_blk.ar_copies(1, 2):
                d.start()
        zero_seen.wait()
        lax.fori_loop(0, L, layer, 0)
        if ctx.ARP:  # the last layer's MoE all-reduce round 2
            allreduce_blk.arp_round2_add()
        allreduce_blk.wait_sends()
        if FUSED:
            if c.NGDN > 0:
                for slot_o in range(NGO):
                    gdn_dma.wait_gdn_out(slot_o)
            for s in range(NS):
                kv_dma.wait_kv_wr(s)
        elif c.NGDN > 0:
            o_last = c.NGDN - 1
            pltpu.make_async_copy(
                gdn_dma.conv_out_src(), O["conv"].at[o_last], scr["st_sem"].at[2]
            ).wait()
            if GCH:
                gdn_dma.rec_drain(o_last)
            else:
                pltpu.make_async_copy(
                    scr["recb"], O["rec"].at[o_last], scr["st_sem"].at[3]
                ).wait()
        cp = pltpu.make_async_copy(W["final_norm"], scr["fn"], scr["misc_sem"])
        cp.start()
        cp.wait()
        out_ref[:MB] = rms_zc(scr["res"][...], scr["fn"][...]).astype(BF16)
        if MB < M:
            out_ref[MB:] = jnp.zeros((M - MB, H), BF16)

    call = pl.pallas_call(
        kernel,
        out_shape=ctx.out_shape,
        in_specs=ctx.in_specs,
        out_specs=ctx.out_specs,
        scratch_shapes=[ctx.scratch[k] for k in ctx.s_names],
        # GDN states update in place: layer o's slice is read (and waited) before it is written
        input_output_aliases=ctx.aliases,
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID if R > 1 else None,
            vmem_limit_bytes=tiles.vmem_mib * 1024 * 1024,
            disable_bounds_checks=not tiles.bounds_checks,
        ),
        interpret=(
            interpret
            if isinstance(interpret, pltpu.InterpretParams)
            else pltpu.InterpretParams()
            if interpret
            else False
        ),
        name="qwen3_5_decode_megakernel",
    )
    return call, ctx.info


def kernel_wide(ctx: MKBuildContext, refs):
    """Wide step body: layer by layer, the mixer phase runs per microbatch on the microbatch's
    rows and the MoE phase once over every row (MoEBlock.moe_wide), so each hit expert's
    weights stream once per layer. Residual rows live in res_all (bf16-exact) across layers;
    hidden in and out are HBM operands copied per microbatch."""
    ctx.bind_refs(refs)
    rings = WeightRings(ctx)
    allreduce_blk = PallasAllReduce(ctx)
    gdn_dma = GDNPoolDMA(ctx)
    kv_dma = KVPoolDMA(ctx)
    gdn_blk = GDNBlock(ctx, rings, gdn_dma)
    att_blk = PagedAttentionBlock(ctx, rings, kv_dma)
    moe_blk = MoEBlock(ctx, rings)

    c, scr, R = ctx.c, ctx.scr, ctx.R
    W, hid_hbm, out_hbm = ctx.W, ctx.hid_ref, ctx.out_ref
    sc, sm, rms_zc = ctx.sc, ctx.sm, ctx.rms_zc
    B, MB, MR, H, L, E, AE = ctx.B, ctx.MB, ctx.MR, ctx.H, ctx.L, ctx.E, ctx.AE
    NS, NBD, NGI, NGO = ctx.NS, ctx.NBD, ctx.NGI, ctx.NGO
    FUSED, ECOPY, SPLIT, GQ, PACK, QV = ctx.FUSED, ctx.ECOPY, ctx.SPLIT, ctx.GQ, ctx.PACK, ctx.QV
    t_block, t_gdn, lay = ctx.t_block, ctx.t_gdn, ctx.lay
    me, rb_hbm = ctx.me, ctx.rb_hbm
    N_MB, BW, BWP, MB_GDN, MB_ATT = ctx.N_MB, ctx.BW, ctx.BWP, ctx.MB_GDN, ctx.MB_ATT
    M_OUT = ctx.c_wide.M
    res_all, xb_all, part_all = scr["res_all"], scr["xb_all"], scr["part_all"]
    assert FUSED and MB == B and B % 16 == 0, (FUSED, MB, B)
    assert not ctx.ARP

    def rows(m):
        return pl.ds(pl.multiple_of(m * B, B), B)

    def ar_slot(seq):
        """Buffer slot of the step's seq-th all-reduce (2 * N_MB per layer: the mixer ones,
        then the MoE ones), and whether it is the slot's first use (no sends to wait for)."""
        if ctx.N_AR_SLOTS == 1:
            return 0, seq == 0
        return lax.rem(seq, 2), seq < 2

    # ------------------------------ layer loop ------------------------------
    def layer(l, carry):
        slot = l % 2
        with sc("norm_in"):
            pltpu.make_async_copy(
                W["smalls"].at[0], scr["sm"].at[slot], scr["sm_sem"].at[slot]
            ).wait()
            if ECOPY:
                o_ec = lay["ecopy"][0]
                pltpu.make_async_copy(
                    W["smalls"].at[0, :, pl.ds(o_ec, E)],
                    scr["ecopy"].at[slot],
                    scr["ec_sem"].at[slot],
                ).wait()

            @pl.when(l + 1 < L)
            def _():
                pltpu.make_async_copy(
                    W["smalls"].at[l + 1],
                    scr["sm"].at[1 - slot],
                    scr["sm_sem"].at[1 - slot],
                ).start()
                if ECOPY:
                    o_ec = lay["ecopy"][0]
                    pltpu.make_async_copy(
                        W["smalls"].at[l + 1, :, pl.ds(o_ec, E)],
                        scr["ecopy"].at[1 - slot],
                        scr["ec_sem"].at[1 - slot],
                    ).start()

        is_att = l % AE == AE - 1
        base_l = (l // AE) * t_block + (l % AE) * t_gdn
        per_mb = jnp.where(is_att, MB_ATT, MB_GDN)

        def mixer(m, carry_m):
            ctx.bind_microbatch(m)
            with sc("norm_in"):
                scr["res"][...] = res_all[rows(m)].astype(F32)
                scr["xb"][...] = rms_zc(scr["res"][...], sm(slot, "in_norm")).astype(BF16)
            ars, first = ar_slot(l * (2 * N_MB) + m)
            g0 = base_l + m * per_mb

            def att_branch():
                att_blk.init_worklist()
                att_blk.init_rope()
                if ctx.AR0S and ctx.comm:
                    att_blk.attention(l, slot, g0, ar0=allreduce_blk.ar0_spec(slot, first, ars))
                    allreduce_blk.ar0_finish(ars, first)
                    return
                allreduce_blk.add_residual(
                    allreduce_blk.allreduce(ars, att_blk.attention(l, slot, g0), first)
                )

            def gdn_branch():
                if ctx.AR0S and ctx.comm:
                    gdn_blk.gdn(l, slot, g0, ar0=allreduce_blk.ar0_spec(slot, first, ars))
                    allreduce_blk.ar0_finish(ars, first)
                    return
                allreduce_blk.add_residual(
                    allreduce_blk.allreduce(ars, gdn_blk.gdn(l, slot, g0), first)
                )

            if AE == 1:
                att_branch()
            else:
                lax.cond(is_att, att_branch, gdn_branch)
            with sc("moe_router"):
                xb_all[rows(m)] = rms_zc(scr["res"][...], sm(slot, "post_norm")).astype(BF16)
                res_all[rows(m)] = scr["res"][...].astype(BF16)
            return carry_m

        lax.fori_loop(0, N_MB, mixer, 0)
        moe_blk.moe_wide(l, slot, base_l + N_MB * per_mb)

        def moe_ar(m, carry_m):
            ars, first = ar_slot(l * (2 * N_MB) + N_MB + m)
            tot = allreduce_blk.allreduce(ars, part_all[rows(m)], first)
            with sc("ar_res"):
                res_all[rows(m)] = rnd(res_all[rows(m)].astype(F32) + tot[0:B]).astype(BF16)
            return carry_m

        lax.fori_loop(0, N_MB, moe_ar, 0)
        return carry

    # ------------------------------ step ------------------------------
    scr["hitv"][...] = jnp.zeros((MR, 128), I32)
    zero_seen = pltpu.make_async_copy(
        scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"]
    )
    zero_seen.start()
    for i in range(NBD):
        rings.fetch(jnp.int32(i))
    pltpu.make_async_copy(
        W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]
    ).start()
    if ECOPY:
        o_ec = lay["ecopy"][0]
        pltpu.make_async_copy(
            W["smalls"].at[0, :, pl.ds(o_ec, E)],
            scr["ecopy"].at[0],
            scr["ec_sem"].at[0],
        ).start()
    if ctx.forced_routing:  # the first 128-row routing chunk; moe_wide chains the rest
        pltpu.make_async_copy(
            rb_hbm.at[0, pl.ds(0, 128)], scr["rb"].at[0], scr["rb_sem"].at[0]
        ).start()
    hid_in = pltpu.make_async_copy(
        hid_hbm, res_all.at[pl.ds(0, M_OUT)], scr["io_sem"].at[0]
    )
    hid_in.start()
    if R > 1:
        barrier = pltpu.get_barrier_semaphore()
        for off in range(1, R):
            pl.semaphore_signal(
                barrier,
                1,
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(barrier, R - 1)
    scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
    scr["knew_o"][...] = jnp.zeros((MR, ctx.HD), FP8)
    scr["vnew_o"][...] = jnp.zeros((MR, ctx.HD), FP8)
    for s in range(NS):
        scr["kv_rd_pend"][s] = jnp.int32(0)
        scr["kv_wr_pend"][s] = jnp.int32(0)
    if c.NGDN > 0:
        for slot_i in range(NGI):
            scr["gdn_in_pend"][slot_i] = jnp.int32(0)
        for slot_o in range(NGO):
            scr["gdn_out_mode"][slot_o] = jnp.int32(0)
        scr["gdn_out_page"][...] = jnp.zeros(scr["gdn_out_page"].shape, FP8)
    if SPLIT and not GQ:
        scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)
    if PACK or QV:
        scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
    xb_all[...] = jnp.zeros(xb_all.shape, BF16)
    hid_in.wait()
    if BW > M_OUT:
        res_all[M_OUT:BW, :] = jnp.zeros((BW - M_OUT, H), BF16)
    zero_seen.wait()
    lax.fori_loop(0, L, layer, 0)
    allreduce_blk.wait_sends()
    if c.NGDN > 0:
        for slot_o in range(NGO):
            gdn_dma.wait_gdn_out(slot_o)
    for s in range(NS):
        kv_dma.wait_kv_wr(s)
    cp = pltpu.make_async_copy(W["final_norm"], scr["fn"], scr["misc_sem"])
    cp.start()
    cp.wait()
    for m in range(N_MB):
        n = min(B, M_OUT - m * B)
        if n <= 0:
            break
        scr["xb"][...] = rms_zc(res_all[m * B:(m + 1) * B, :].astype(F32), scr["fn"][...]).astype(BF16)
        out_cp = pltpu.make_async_copy(
            scr["xb"].at[pl.ds(0, n)], out_hbm.at[pl.ds(m * B, n)], scr["io_sem"].at[1]
        )
        out_cp.start()
        out_cp.wait()


def make_step(
    mesh,
    c: MKConfig,
    tiles: Tiles | None = None,
    *,
    interpret=False,
    forced_routing=False,
    skip=frozenset(),
    scopes=False,
):
    """shard_map decode step: (pos [B], hidden [R*M, H] (rank-replicated rows), weights, states,
    [rbias [R*L, M, E]]) -> (hidden_out [R*M, H], states). Weights carry a leading rank axis
    [R, ...]; states are rank-folded [R * N0, ...] (see `fold`) so that each shard is exactly the
    kernel operand and the in-place state update needs no copy. Both are sharded over "tp".
    T > 1 (verify): the returned conv / rec are the per-token outputs of kernel_out_shapes (feed
    accept_states' result back), and every row's new K/V lands in its sequence's cache at pos."""
    call, info = build(
        c,
        tiles,
        interpret=interpret,
        forced_routing=forced_routing,
        skip=skip,
        scopes=scopes,
        pool_gather=False,
    )

    def local(pos, hidden, weights, states, *rb):
        w = [
            tuple(x[0] for x in weights[k])
            if isinstance(weights[k], tuple)
            else weights[k][0]
            for k in W_NAMES
        ]
        s = [states[k] for k in S_NAMES]
        hid, conv, rec, knew, vnew = call(pos, hidden, *w, *s, *rb)
        kc, vc = states["kc"], states["vc"]
        if c.NS > 16:  # one scatter per cache (2 * NS dynamic-update-slices cost ~1.6 us each)
            bi = jnp.arange(c.B) if c.T == 1 else jnp.arange(c.B) // c.T
            kw = dict(indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds")
            kc = kc.at[:, bi, pos].set(knew[:, :c.B].astype(BF16), **kw)
            vc = vc.at[:, bi, pos].set(vnew[:, :c.B].astype(BF16), **kw)
            return hid, {"conv": conv, "rec": rec, "kc": kc, "vc": vc}
        for s in range(c.NS):
            b0 = s * c.T
            upd = (0, s, pos[b0], 0)
            kc = lax.dynamic_update_slice(kc, knew[:, None, b0:b0 + c.T].astype(BF16), upd)
            vc = lax.dynamic_update_slice(vc, vnew[:, None, b0:b0 + c.T].astype(BF16), upd)
        return hid, {"conv": conv, "rec": rec, "kc": kc, "vc": vc}

    def step(pos, hidden, weights, states, *rb):
        ws = {
            k: tuple(P("tp") for _ in weights[k])
            if isinstance(weights[k], tuple)
            else P("tp")
            for k in weights
        }
        ss = {k: P("tp") for k in states}
        in_specs = (P(), P("tp"), ws, ss) + ((P("tp"),) if forced_routing else ())
        return jax.shard_map(
            local, mesh=mesh, in_specs=in_specs, out_specs=(P("tp"), ss), check_vma=False
        )(pos, hidden, weights, states, *rb)

    return step, info


__all__ = [
    "ABLATIONS",
    "BF16",
    "COLLECTIVE_ID",
    "CONV_STAGE_MIN_B",
    "DEFAULT_TILES",
    "DENSE",
    "DMM_UNROLL",
    "EXPERT",
    "F32",
    "FP8",
    "FULL",
    "FULL_EP",
    "GQ_PACK",
    "I32",
    "MKConfig",
    "NN",
    "NT",
    "O_NAMES",
    "S_NAMES",
    "Spec",
    "TINY",
    "TINY_EP",
    "TINY_TILES",
    "TOPK_LANES",
    "TOPK_NL",
    "Tiles",
    "U8",
    "U32",
    "VERIFY_TILES",
    "W_NAMES",
    "WIDE_TILES",
    "accept_states",
    "bswitch",
    "build",
    "default_tiles",
    "encode_ecopy",
    "fold",
    "kernel_out_shapes",
    "make_step",
    "plan",
    "pool_alias",
    "rnd",
    "rup",
    "smalls_layout",
    "state_shapes",
    "topk_rows_on_lanes",
    "unfold",
    "weight_shapes",
    # Modular building blocks
    "MKBuildContext",
    "DenseWeightRing",
    "ExpertWeightRing",
    "WeightRings",
    "PallasAllReduce",
    "GDNPoolDMA",
    "KVPoolDMA",
    "GprRecIn",
    "GDNBlock",
    "PagedAttentionBlock",
    "MoEBlock",
]
