# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Batched DSA core of the glm52_decode_mk megakernel.

Ported from G1's core: glm5_mk_dcp/dsa_batched.py on hning/glm52-mk-g1-dsa-spike @ 0a73f68bc (md5 2085f8d7, the
G1 verdict's frozen file; G1 design kernel_work/glm52-decode-mk/gates/g1/design.md). The algorithm, the scratch
geometry and the exchange protocol are unchanged. Changes for the composed kernel (WHITEBOARD.v1 §4, §7, §11.2):
  * k_scale is a per-layer runtime value (`E.ks()`, the layer's smalls row) instead of a static float, applied
    unconditionally (x 1.0 is exact), so one program serves checkpoints with and without KV scales;
  * the index-K scan resolves pages through its own block table `E.bti` (the indexer cache's page count may differ
    from the latent cache's: production resolves virtual ordinals `mod` each cache's own page count);
  * `ALIASED` names the layer-transient buffers the kernel places on the idle expert ring (`epool` views); the
    kernel allocates them, this module only declares them;
  * debug export of the pre-resolution union lists (`E.dbg_uv`) for the per-row top-k contract;
  * `tiles.recip`: "exact" 1 / l (the default and the production build since the stage-A fix pass) | "approx" (the
    served sparse-attention kernel's EUP reciprocal; interpret lowers approx=True to a bf16 reciprocal);
  * fault controls added for the composed kernel's oracle: causal_mask, page_off_by_one, kv_read_before_write, topk_swap,
    scale_twice (WHITEBOARD.v1 §11.2), and an F-DELAY hook `E.delay(group, round)` in the count exchange.

Stage-B lane L1 (WHITEBOARD.v3 §6.1; branch hning/glm52-decode-mk-b-l1-threshold), each a Tiles flag whose default
is the stage-A program (the default build lowers to the identical Mosaic module):
  * thr_unroll  (L1-U) the threshold count loop takes thr_unroll tiles per iteration (per-splitter tree sums; exact);
  * thr_merge   (L1-M) one count exchange per round: every 8-row group's counts in ONE (8, 128) vreg, group g in
                lanes [32 g, 32 g + K) -- 15 x 4 KiB per round instead of NG x 15 x 4 KiB; same digits (exact);
  * thr_switch >= pool compiles only the 4-bit body (L1-W: 8 rounds at every Sigma np);
  * rec_thr / thr_given / union_given: timing-only backing arms (record b_row T / P / allowance and the resolved
    union table in one build; load them and skip threshold() [and the unions] in another; outputs bit-identical).

The original module docstring follows.

Batched DSA core for the glm5_mk_dcp megakernel (G1 design: kernel_work/glm52-decode-mk/gates/g1/design.md).

Replaces the per-request / per-row index path (scan_request, topk2 local + global, candidate-key AG, kloc
fallback, scalar union, resolve) and the per-request attention loop with, per fresh layer:

  scan    one continuous K ring across all requests (SD slots x NPG pages), production head-sum order,
          keys stored as one aligned (8, 128) tile per page: sublane 4c + q = row q, positions 256p + 128c + lane
  fix     the step's own new index records scored by one small matmul and patched into their tiles
  thr     exact distributed radix threshold: per round every rank counts #(key >= splitter) per row for the
          2^DW - 1 splitters of the next digit and all-gathers the counts (one (8, 128) vreg per 8-row group);
          every rank sums them itself.  The last round also counts the bracket's ends (k = 0 and k = 2^DW),
          so each rank's share of the tied keys follows from the per-rank counts without another round.
          Tie rule = production's two-level rule = global order (key desc, rank asc, local index asc).
  union   per request: 4-row membership bits per position -> one counting extraction -> own_uv entries
          (local index | row bits << MB_SH), ascending
then on every layer one all-request gather ring (GR granules x 128 entries, manual unroll, one byte-count
wait per granule per stream) and a per-granule attention step (rope quarter realigned per entry: 4 nope +
1 rope matmul; online softmax; PV in bf16 or f32; k_scale as the served kernel applies it).

Everything is a closure over the megakernel's refs; `make(E)` takes a namespace E built inside kernel().
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
U32 = jnp.uint32
U8 = jnp.uint8
INT_MIN = np.int32(-(2**31))
INT_MAX = np.int32(2**31 - 1)
NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))

FAULTS = (
    "",
    "digit",
    "tierank",
    "nofix",
    "kscale",
    "page",
    "ropeseg",
    "sharedsem",
    # composed-kernel oracle faults (glm_ref names where the semantics match)
    "causal_mask",
    "page_off_by_one",
    "kv_read_before_write",
    "scale_twice",
    "topk_swap",
    # delay-exposed race control: count-exchange payloads read before their receives are waited
    "f_cx_earlyread",
    # path-activation controls for the forced-path variants (WHITEBOARD.v1 §11.2 C-THR2 / C-WAVE / C-WRAP):
    # each perturbs only its path, so n/a at a build that never takes it and RED where it does
    "f_thr2",  # digit + 1, 2-bit threshold rounds only (Sigma np > thr_switch)
    "f_wave",  # digit + 1, waves after the first only (rows of a wave starting at row > 0)
    "f_wrap",  # granules r >= GR read ring slot (r + 1) % GR (the gather ring has wrapped)
    # L4 (stage B, a2a_lse="tile"): the lse of row t is written at lane t + 1 of the per-peer lse tile
    "f_lse_tile_row",
    # stage-B L2 `gearly` control: the early (previous-layer) issue reads layer l - 1's caches -> RED
    "f_early_stale",
    # V16 (stageB/v16/DESIGN-ADDENDUM.md, VR > 1 builds only): tie allowance in PHYSICAL rank order instead of the
    # virtual order v(r) = VR * (r % D) + r // D; replicas on half 1 use the other half's page parity; the scan / fix /
    # union positions use the DCP-R rule (G = ILV * R, shard = rank) on the DCP-D cache
    "f_vorder",
    "f_replica_half",
    "f_v16_owner",
)
# layer-transient buffers: produced and consumed between layer entry and routing (WHITEBOARD.v1 §5.3 / §7.1)
ALIASED = (
    "b_keys",
    "b_memb",
    "b_pre",
    "b_xacc",
    "b_xcc",
    "g_nope",
    "g_rope",
    "kvb",
    "qb",
    "qrb",
    "b_sbuf",
    "att_acc",
)
# Fault builds that also place the IndexShare table `own_uv` in the alias window (kernel.aliased_names). There
# `own_uv` is a pool_alias view with an explicit (8, 128) tiling, so every DMA window into it must start on an 8-row
# tile (Mosaic verifies this on libtpu 0.0.47); udma then copies whole 8-row granules and `own_u` gets the padded
# row count. Clean builds keep `own_uv` as native scratch and are unchanged.
UV_ALIAS_FAULTS = ("f_stale_is_alias", "f_alias_poison_red")


def uv_rows(g, tiles):
    """Rows of own_u / own_uv: UR2, rounded up to the 8-row tile in the UV_ALIAS_FAULTS builds."""
    return -(-g["UR2"] // 8) * 8 if tiles.fault in UV_ALIAS_FAULTS else g["UR2"]


def shl(x, n):
    return lax.shift_left(x, jnp.asarray(n, x.dtype))


def shr(x, n):
    return lax.shift_right_logical(x, jnp.asarray(n, x.dtype))


def geometry(cfg, tiles, max_pages, bmax):
    """Static sizes of the batched core."""
    TMAX = cfg.TMAX
    BMAX = bmax or TMAX
    KT = cfg.KT
    KR_ = KT // 128
    TR = max(8, (TMAX + 7) // 8 * 8)  # per-row state rows
    NG = TR // 8  # 8-row threshold groups
    UR_ = TMAX * KR_ + BMAX  # union rows (sum of ceil(nu / 128)) <= this
    XKC = tiles.xkc
    UR2 = UR_ + XKC // 128  # + the extraction's last-chunk spill
    MEMBR = max(
        128, (2 * max_pages + 127) // 128 * 128
    )  # membership rows (2 per page), 128-row blocks
    POOL = tiles.pool
    NPG = tiles.scan_np
    NBLK = POOL // NPG + BMAX + 1  # scan blocks per wave (upper bound)
    PB = max(
        1, int(math.ceil(math.log2(max(2, max_pages * cfg.page))))
    )  # position bits (tie bound)
    QN = cfg.QMAX * cfg.NH
    return dict(
        TMAX=TMAX,
        BMAX=BMAX,
        KT=KT,
        KR_=KR_,
        TR=TR,
        NG=NG,
        UR_=UR_,
        UR2=UR2,
        XKC=XKC,
        MEMBR=MEMBR,
        POOL=POOL,
        NPG=NPG,
        SD=tiles.scan_sd,
        NBLK=NBLK,
        PB=PB,
        GR=tiles.gring,
        QN=QN,
    )


def scratch(cfg, tiles, max_pages, bmax):
    """The batched core's scratch (added to the megakernel's)."""
    g = geometry(cfg, tiles, max_pages, bmax)
    R, pg, KL = cfg.R, cfg.page, cfg.KL
    IR4 = cfg.IREC // 4
    qdt = FP8 if tiles.dot == "fp8" else BF16
    return {
        # scan
        "b_kring": pltpu.VMEM((g["SD"], g["NPG"], IR4, 4, pg), U8),
        "b_kring_sem": pltpu.SemaphoreType.DMA((g["SD"],)),
        "b_keys": pltpu.VMEM((g["POOL"], 8, 128), I32),
        "b_qblk": pltpu.VMEM((cfg.IH * 8, 128), qdt),
        "b_wb": pltpu.VMEM((cfg.IH, 4, pg), F32),
        "b_lim": pltpu.VMEM((4, pg), I32),  # per-request head weights / row limits
        "b_qblk4": pltpu.VMEM((cfg.IH * 4, 128), qdt),  # scan q block, 4 rows per head
        "b_knew": pltpu.VMEM((128, 128), FP8),
        "b_snew": pltpu.VMEM((8, 128), F32),
        # threshold: per (group, parity) send / receive buffers and semaphores
        "b_csend": pltpu.VMEM((g["NG"], 2, 8, 128), I32),
        "b_crecv": pltpu.VMEM((g["NG"], 2, R, 8, 128), I32),
        "b_csem_s": pltpu.SemaphoreType.DMA((g["NG"], 2)),
        "b_csem_r": pltpu.SemaphoreType.DMA((g["NG"], 2)),
        # per-row state (lane-broadcast): 0 mode, 1 prefix, 2 T, 3 P, 4 allowance
        "b_row": pltpu.VMEM(
            (6 if getattr(tiles, "thr_early", False) else 5, g["TR"], 128), I32
        ),
        # union extraction
        "b_memb": pltpu.VMEM((g["MEMBR"], 128), I32),
        "b_pre": pltpu.VMEM((g["MEMBR"], 128), I32),
        "b_xacc": pltpu.VMEM((g["XKC"], 384), F32),
        "b_xcc": pltpu.VMEM((2, 128, 128), F32),
        "own_u": pltpu.SMEM((uv_rows(g, tiles), 128), I32),
        "own_uv": pltpu.VMEM((uv_rows(g, tiles), 128), I32),
        # per request: 0 nu, 1 uo, 2 nuo, 3 np (local pages), 4 pbase, 5 seq, 6 nold, 7 t0, 8 nq, 9 p0,
        # 10 valid, 11 npmax (rank-0 pages: the wave partition), 12 needs tie bound
        "b_req": pltpu.SMEM((g["BMAX"], 16), I32),
        # per granule (union row): request
        "b_greq": pltpu.SMEM((g["UR2"],), I32),
        # scalars: 0 U (union rows), 1 n waves, 2 nblk (current wave), 3 any radix row (current wave)
        "b_st": pltpu.SMEM((16,), I32),
        "b_wave": pltpu.SMEM((g["BMAX"] + 2,), I32),
        "b_blk": pltpu.SMEM((2, g["NBLK"]), I32),
        "b_grp": pltpu.SMEM((2, g["NG"]), I32),
        **(
            {"b_tg": pltpu.SMEM((2, g["NG"]), I32)}
            if getattr(tiles, "thr_early", False)
            else {}
        ),  # L1-E: done, parity base
        # gather ring + attention
        "g_nope": pltpu.VMEM((g["GR"] * 128, 128), U32),
        "g_rope": pltpu.VMEM((g["GR"] * 128, 128), U32),
        "g_sem": pltpu.SemaphoreType.DMA((g["GR"], 2)),
        "kvb": pltpu.VMEM((2, 128, KL), BF16),
        "b_sbuf": pltpu.VMEM((2, g["QN"], 128), F32),
        "qb": pltpu.VMEM((g["QN"], KL), BF16),
        "qrb": pltpu.VMEM((g["QN"], 128), BF16),
        "att_acc": pltpu.VMEM((g["QN"], KL), F32),
        "m_s": pltpu.VMEM((g["QN"], 1), F32),
        "l_s": pltpu.VMEM((g["QN"], 1), F32),
    }


def extract_union(
    keys_ref, nbk, out_ref, row0, pre_ref, acc_ref, cc_ref, kc, mb_sh, max_rows=None
):
    """Counting extraction (topk2.extract_payload's math, one payload plane): every position with
    key > 0 over the first nbk 128-row blocks of keys_ref (position = row * 128 + lane), ascending, as
    out_ref rows [row0, row0 + ceil(n / 128)): position | key << mb_sh (keys are < 2^(31 - mb_sh)), -1 past
    the count (the last chunk may write up to kc / 128 - 1 rows beyond).  Returns n."""
    ri = lax.broadcasted_iota(I32, (128, 128), 0)
    li = lax.broadcasted_iota(I32, (128, 128), 1)
    ones8 = jnp.ones((8, 128), BF16)
    upper = (ri <= li).astype(BF16)
    cc_ref[0] = jnp.zeros((128, 128), F32)

    def sel_blk(b, c):
        r0 = pl.multiple_of(b * 128, 128)
        kb = keys_ref[pl.ds(r0, 128), :]
        selb = (kb > 0).astype(F32).astype(BF16)
        cc_ref[0, pl.ds(b, 1), :] = lax.dot_general(
            ones8, selb, NT, preferred_element_type=F32
        )[0:1]
        pre_ref[pl.ds(r0, 128), :] = lax.dot_general(
            selb, upper, NN, preferred_element_type=F32
        ).astype(I32)
        return c

    lax.fori_loop(0, nbk, sel_blk, 0)
    C = cc_ref[0]
    n_sel = jnp.sum(C).astype(I32)
    W = lax.dot_general(C.astype(BF16), upper, NN, preferred_element_type=F32)
    tot = W[:, 127:128]
    th = jnp.floor(tot * (1.0 / 128))
    tl = tot - 128 * th
    Ls = (li < ri).astype(BF16)
    off = 128 * lax.dot_general(
        Ls,
        jnp.broadcast_to(th, (128, 128)).astype(BF16),
        NN,
        preferred_element_type=F32,
    ) + lax.dot_general(
        Ls,
        jnp.broadcast_to(tl, (128, 128)).astype(BF16),
        NN,
        preferred_element_type=F32,
    )
    CI = W + off
    CE = CI - C
    cc_ref[0] = CE
    cc_ref[1] = CI
    valid = ri < nbk

    def chunk(ci, c):
        c0 = ci * kc
        kr = (c0 + lax.broadcasted_iota(I32, (kc, 1), 0)).astype(F32)
        r_first = jnp.sum(((cc_ref[1] <= c0) & valid).astype(I32))
        r_end = jnp.sum(((cc_ref[0] < c0 + kc) & valid).astype(I32))
        b_lo = r_first // 128
        b_hi = jnp.minimum(jnp.maximum((r_end + 127) // 128, b_lo), nbk)

        def body(b, c2):
            ce_r = cc_ref[0, pl.ds(b, 1), :]
            ci_r = cc_ref[1, pl.ds(b, 1), :]
            oht = ((ce_r <= kr) & (kr < ci_r)).astype(F32).astype(BF16)  # (kc, 128)
            r0 = pl.multiple_of(b * 128, 128)
            pre = pre_ref[pl.ds(r0, 128), :].astype(F32).astype(BF16)
            ce_c = jnp.transpose(jnp.broadcast_to(ce_r, (128, 128)))
            ceh = jnp.floor(ce_c * (1.0 / 128))
            tail = jnp.where(
                li == 0,
                b.astype(F32),
                jnp.where(
                    li == 1,
                    ri.astype(F32),
                    jnp.where(li == 2, ceh, jnp.where(li == 3, ce_c - 128 * ceh, 0.0)),
                ),
            ).astype(BF16)
            plane = (keys_ref[pl.ds(r0, 128), :] & 0xFF).astype(F32).astype(BF16)
            parts = (
                lax.dot_general(oht, pre, NN, preferred_element_type=F32),
                lax.dot_general(oht, tail, NN, preferred_element_type=F32),
                lax.dot_general(oht, plane, NN, preferred_element_type=F32),
            )

            @pl.when(
                b == b_lo
            )  # first block assigns: no zero fill, no read-modify-write
            def _():
                for i_, v_ in enumerate(parts):
                    acc_ref[:, 128 * i_ : 128 * (i_ + 1)] = v_

            @pl.when(b > b_lo)
            def _():
                for i_, v_ in enumerate(parts):
                    acc_ref[:, 128 * i_ : 128 * (i_ + 1)] += v_

            return c2

        lax.fori_loop(b_lo, b_hi, body, 0)
        RT = acc_ref[...]
        rr = 128 * RT[:, 128:129] + RT[:, 129:130]
        ce = 128 * RT[:, 130:131] + RT[:, 131:132]
        lane = jnp.sum((RT[:, 0:128] <= kr - ce).astype(F32), axis=1, keepdims=True)
        live = kr < n_sel.astype(F32)
        lsel = lax.broadcasted_iota(I32, (kc, 128), 1).astype(F32) == lane
        key = jnp.sum(jnp.where(lsel, RT[:, 256:384], 0.0), axis=1, keepdims=True)
        ent = jnp.where(
            live, (rr * 128 + lane).astype(I32) | shl(key.astype(I32), mb_sh), -1
        )  # (kc, 1)
        entr = jnp.transpose(jnp.broadcast_to(ent, (kc, 128)))[0:1]  # (1, kc)
        for q in range(kc // 128):
            out_ref[pl.ds(row0 + ci * (kc // 128) + q, 1), :] = entr[
                :, q * 128 : (q + 1) * 128
            ]
        return c

    nch = (n_sel + kc - 1) // kc
    if (
        max_rows is not None
    ):  # never write past the output (only a corrupt selection gets here)
        cap = jnp.maximum(max_rows - kc // 128, 0) // (kc // 128)
        nch = jnp.minimum(nch, cap)
        n_sel = jnp.minimum(n_sel, nch * kc)
    lax.fori_loop(0, nch, chunk, 0)
    return n_sel


def make(E):
    """E: namespace with cfg, tiles, scr, consts and helper closures (see kernel.build)."""
    cfg, tiles, scr = E.cfg, E.tiles, E.scr
    g = geometry(cfg, tiles, E.max_pages, E.BMAX)
    R, me = cfg.R, E.me
    TMAX, BMAX, TR, NG, UR2 = g["TMAX"], g["BMAX"], g["TR"], g["NG"], g["UR2"]
    KT, POOL, NPG, SD, _NBLK, PB, GR, QN, XKC = (
        g["KT"],
        g["POOL"],
        g["NPG"],
        g["SD"],
        g["NBLK"],
        g["PB"],
        g["GR"],
        g["QN"],
        g["XKC"],
    )
    pg, lg_pg = cfg.page, int(math.log2(cfg.page))
    assert pg == 256 and cfg.QMAX == 4, (
        pg,
        cfg.QMAX,
    )  # (8, 128) key tile = 4 rows x 2 chunks
    ILV, G_ = cfg.ILV, cfg.ILV * R
    # V16 (VR > 1): the cache is DCP-D; rank me holds shard mk = me % D and is replica mh = me // D. Request b's local
    # page j8 of that shard is this rank's iff (j8 + b) % VR == mh: its pool tile p is page j8 = VR * p + hb, hb =
    # (mh + b) % VR (req[b, 13]). Tile order = local-index order, so tile-space positions / counts compare exactly.
    VR, D_ = cfg.VR, cfg.D
    if VR > 1:
        if tiles.fault == "f_v16_owner":  # FAULT: the DCP-R position rule on the DCP-D cache
            G_, mk, mh = ILV * R, me, me // D_
        else:
            G_, mk, mh = ILV * D_, me % D_, me // D_

        def l8_of_tile(t, hb):
            """tile-space position t (tile << 8 | slot) of a request with half offset hb -> shard-local index."""
            return (VR * (t >> lg_pg) + hb) * pg + (t & (pg - 1))

        def gpos_of_l8(l8):
            return (l8 // ILV) * G_ + mk * ILV + l8 % ILV
    IH, IDIM, NH, KL, HR = cfg.IH, cfg.IDIM, cfg.NH, cfg.KL, cfg.HR
    assert IH * 8 <= 512 and NH <= 128
    MB_SH, SRC_MASK = E.MB_SH, E.SRC_MASK
    QA, QH = E.QA, E.QH
    rmeta, qmeta, bt, rvec = E.rmeta, E.qmeta, E.bt, E.rvec
    bti = E.bti  # index-K cache pages (its own modulus)
    MAXP = E.max_pages
    skip = E.skip
    fault = tiles.fault
    # kernel.FAULTS validates the name; faults outside FAULTS belong to the host kernel and are no-ops here
    PSH = (
        1 if fault == "page_off_by_one" else 0
    )  # FAULT: virtual page + 1 in the scan and the gather
    sm_scale = E.sm_scale
    Phases, csel, mxu, _fp8_decode_hw, mono_key, pack2, unpack_lo, unpack_hi = (
        E.Phases,
        E.csel,
        E.mxu,
        E.fp8_decode_hw,
        E.mono_key,
        E.pack2,
        E.unpack_lo,
        E.unpack_hi,
    )
    req = scr["b_req"]
    st = scr["b_st"]
    row = scr["b_row"]
    sub8 = lax.broadcasted_iota(I32, (8, 128), 0)
    lane8 = lax.broadcasted_iota(I32, (8, 128), 1)
    NEG = jnp.finfo(F32).min
    sub128 = lax.broadcasted_iota(I32, (128, 1), 0)
    UNR = max(1, tiles.gunroll)
    MAXP_OK = E.max_pages <= 256
    SCAN_UNROLL = int(
        tiles.scan_unroll
    )  # per-lane tile counts <= 256: exact bf16 MXU lane sums
    LSE_TILE = getattr(tiles, "a2a_lse", "lanes") == "tile"
    HPAD = max(
        8, -(-cfg.HR // 8) * 8
    )  # lse tile rows per owner (kernel.build: the same rule)
    THR_U = max(
        1, int(tiles.thr_unroll)
    )  # L1-U: count-loop tiles per iteration (1 = stage A)
    THR_SWAR = bool(
        getattr(tiles, "thr_swar", False)
    )  # L1-S: 4-bit counts from packed 8-bit digit histograms
    SW_CH = 248  # SWAR flush period (tiles): an 8-bit counter gains <= 1 per tile
    DSEND = bool(
        getattr(tiles, "thr_dsend", False)
    )  # L1-D: count-exchange send waits deferred by two rounds
    EARLY = bool(
        getattr(tiles, "thr_early", False)
    )  # L1-E: a group stops once #(key >= its prefix) == KT
    MSLOT = 32  # L1-M: lanes per group in the merged count payload
    # L1-M dispatch: merged exchange when the wave's rank-0 page count <= thr_merge (0: never = stage A; >= POOL:
    # always, the per-group form is not compiled). L1-W: thr_switch >= POOL compiles only the 4-bit body.
    MERGE = (
        "never"
        if tiles.thr_merge <= 0
        else ("always" if tiles.thr_merge >= POOL else "dispatch")
    )
    if MERGE != "never":
        assert NG * MSLOT <= 128, (
            "merged count payload holds <= 4 groups (TMAX <= 32)",
            NG,
        )
        assert not DSEND, (
            "thr_dsend is implemented for the per-group (pipelined) exchange only"
        )
        assert not EARLY, (
            "thr_early is implemented for the per-group (pipelined) exchange only"
        )
    ONLY4 = tiles.thr_switch >= POOL
    THR_GIVEN = bool(tiles.thr_given) or bool(
        tiles.union_given
    )  # timing-only backing arms (WHITEBOARD.v3 §6.1)
    UNION_GIVEN = bool(tiles.union_given)

    def T_():
        return E.scal[0]

    def B_():
        return E.scal[1]

    def dup4(x4):
        """(4, 128) rows q -> (8, 128): row q at sublanes q and q + 4."""
        return jnp.concatenate([x4, x4], axis=0)

    def rows4(k, t0):
        """per-row state field k of rows t0..t0+3 -> (8, 128) in key-tile layout."""
        return dup4(row[k, pl.ds(t0, 4), :])

    # ------------------------------------------------------------------ step init
    def step_init():
        """Per-step request geometry (identical on every rank except the rank-local lengths), the per-row
        modes and the wave partition of the key pool."""
        B = B_()
        T = T_()

        def per_req(b, c):
            t0 = qmeta[b, 0]
            nq = qmeta[b, 1]
            seq = qmeta[b, 2]
            ok = jnp.logical_and(b < B, qmeta[b, 3] > 0)
            if VR > 1:
                return per_req_v16(b, c, t0, nq, seq, ok)
            nloc = (seq // G_) * ILV + jnp.clip(seq % G_ - ILV * me, 0, ILV)
            nl0 = (seq // G_) * ILV + jnp.clip(seq % G_, 0, ILV)
            p0 = seq - nq
            nold = (p0 // G_) * ILV + jnp.clip(p0 % G_ - ILV * me, 0, ILV)
            vals = {
                3: jnp.where(ok, (nloc + pg - 1) >> lg_pg, 0),
                5: seq,
                6: nold,
                7: t0,
                8: jnp.where(ok, nq, 0),
                9: p0,
                10: ok.astype(I32),
                11: jnp.where(ok, (nl0 + pg - 1) >> lg_pg, 0),
                0: 0,
                1: 0,
                2: 0,
                4: 0,
                12: 0,
            }
            for k_, v_ in vals.items():
                req[b, k_] = jnp.asarray(v_, I32)
            return c

        def per_req_v16(b, c, t0, nq, seq, ok):
            """V16: shard mk's local counts, this replica's tile count / half offset, nold in tile space."""
            nloc = (seq // G_) * ILV + jnp.clip(seq % G_ - ILV * mk, 0, ILV)
            nl0 = (seq // G_) * ILV + jnp.clip(seq % G_, 0, ILV)
            p0 = seq - nq
            nold8 = (p0 // G_) * ILV + jnp.clip(p0 % G_ - ILV * mk, 0, ILV)
            hb = (mh + b) % VR
            if fault == "f_replica_half":  # FAULT: half-1 replicas take half 0's pages
                hb = jnp.where(mh == 1, hb ^ 1, hb)
            np8 = (nloc + pg - 1) >> lg_pg
            j8n = nold8 >> lg_pg
            nold_t = ((j8n + VR - 1 - hb) // VR) * pg + jnp.where(
                (j8n + VR - hb) % VR == 0, nold8 & (pg - 1), 0
            )
            vals = {
                3: jnp.where(ok, (np8 + VR - 1 - hb) // VR, 0),
                5: seq,
                6: nold_t,
                7: t0,
                8: jnp.where(ok, nq, 0),
                9: p0,
                10: ok.astype(I32),
                11: jnp.where(ok, (((nl0 + pg - 1) >> lg_pg) + VR - 1) // VR, 0),
                0: 0,
                1: 0,
                2: 0,
                4: 0,
                12: 0,
                13: hb,
            }
            for k_, v_ in vals.items():
                req[b, k_] = jnp.asarray(v_, I32)
            return c

        lax.fori_loop(0, BMAX, per_req, 0)
        # per-row modes: 0 inactive, 1 short (pos + 1 <= KT: every valid key), 2 radix
        valid = jnp.logical_and(
            rvec[0:TR, 0:1] > 0, lax.broadcasted_iota(I32, (TR, 1), 0) < T
        )
        pos = rvec[0:TR, 1:2]
        mode = jnp.where(valid, jnp.where(pos + 1 <= KT, 1, 2), 0)
        row[0] = jnp.broadcast_to(mode, (TR, 128))

        def zring(i, c):  # once per step: no NaN bytes in a never-written slot
            r8 = pl.multiple_of(i * 64, 64)
            scr["g_nope"][pl.ds(r8, 64), :] = jnp.zeros((64, 128), U32)
            scr["g_rope"][pl.ds(r8, 64), :] = jnp.zeros((64, 128), U32)
            return c

        lax.fori_loop(0, GR * 2, zring, 0)
        if EARLY:  # L1-E: per-group parity base, alternating across calls
            for gi_ in range(NG):
                scr["b_tg"][1, gi_] = jnp.int32(0)
        # waves: consecutive requests whose rank-0 page counts fit the pool (identical on every rank)

        def wv(b, c):
            acc, nw = c
            npm = req[b, 11]
            new = jnp.logical_and(acc + npm > POOL, acc > 0)
            nw2 = jnp.where(new, nw + 1, nw)

            @pl.when(new)
            def _():
                scr["b_wave"][nw2] = b

            return jnp.where(new, npm, acc + npm), nw2

        scr["b_wave"][0] = jnp.int32(0)
        _, nw = lax.fori_loop(0, B, wv, (jnp.int32(0), jnp.int32(0)))
        st[1] = nw + 1
        scr["b_wave"][nw + 1] = B

    # ------------------------------------------------------------------ count exchange
    def cx_start(gi, par):
        if R == 1 or "coll" in skip:
            return
        sr = scr["b_csem_s"].at[gi, par]
        rr = scr["b_csem_r"].at[gi, 0 if fault == "sharedsem" else par]
        for d in range(R):

            @pl.when(me != d)
            def _(d=d):
                pltpu.make_async_remote_copy(
                    src_ref=scr["b_csend"].at[gi, par],
                    dst_ref=scr["b_crecv"].at[gi, par, me],
                    send_sem=sr,
                    recv_sem=rr,
                    device_id=(jnp.int32(d),),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

    def cx_finish(gi, par):
        if R == 1 or "coll" in skip:
            return
        sr = scr["b_csem_s"].at[gi, par]
        rr = scr["b_csem_r"].at[gi, 0 if fault == "sharedsem" else par]
        desc = pltpu.make_async_remote_copy(
            src_ref=scr["b_csend"].at[gi, par],
            dst_ref=scr["b_crecv"].at[gi, par, 0],
            send_sem=sr,
            recv_sem=rr,
            device_id=(me,),
            device_id_type=pl.DeviceIdType.MESH,
        )

        def waits():
            for _ in range(R - 1):
                desc.wait_recv()
            if (
                not DSEND
            ):  # L1-D: the send completions are waited 2 rounds later (cx_wait_send)
                for _ in range(R - 1):
                    desc.wait_send()

        if tiles.trace:
            with jax.named_scope("X_wait"):
                waits()
        else:
            waits()

    def cx_wait_send(gi, par):
        """L1-D: wait this rank's R - 1 sends of (group gi, parity par): before b_csend[gi, par] is rewritten two
        rounds later, and at the end of the threshold (drain). Same semaphore count as the eager form."""
        if R == 1 or "coll" in skip:
            return
        desc = pltpu.make_async_remote_copy(
            src_ref=scr["b_csend"].at[gi, par],
            dst_ref=scr["b_crecv"].at[gi, par, 0],
            send_sem=scr["b_csem_s"].at[gi, par],
            recv_sem=scr["b_csem_r"].at[gi, 0 if fault == "sharedsem" else par],
            device_id=(me,),
            device_id_type=pl.DeviceIdType.MESH,
        )

        def waits():
            for _ in range(R - 1):
                desc.wait_send()

        if tiles.trace:
            with jax.named_scope("X_wsend"):
                waits()
        else:
            waits()

    # ------------------------------------------------------------------ indexer tail (fresh layers)
    def build_qblk(t0):
        """q block of rows t0..t0+3, head-major: rows 8h + q (q < 4), rows 8h + 4..7 zero (production
        reduce_heads order on an (8, 256) accumulator; the shell's scan layout)."""
        blks = []
        for h_ in range(IH):
            v = scr["iqv"][h_, pl.ds(t0, 8), :]
            blks.append(jnp.where(sub8 < 4, v, 0.0))
        q = jnp.concatenate(blks, axis=0)
        scr["b_qblk"][...] = q.astype(FP8).astype(scr["b_qblk"].dtype)

    def head_sum(st_, t0):
        """st_ (8 * IH, N) f32 -> sum_h relu(st_[8h:8h+8]) * w[:, h], sequential over h."""
        wq = scr["wts"][pl.ds(t0, 8), :]
        wq = jnp.where(sub8 < 4, wq, 0.0)
        acc = jnp.maximum(st_[0:8], 0.0) * wq[:, 0:1]
        for h_ in range(1, IH):
            acc = (
                acc + jnp.maximum(st_[h_ * 8 : (h_ + 1) * 8], 0.0) * wq[:, h_ : h_ + 1]
            )
        return acc

    def qdot(a, b):
        return lax.dot_general(*mxu(a, b, tiles), NN, preferred_element_type=F32)

    def scan_req_setup(t0, nq, seq):
        """Per request (first block): q block, lane-broadcast head weights and the per-row position limit
        (min(pos_q, seq - 1), -1 for rows >= nq) -> VMEM, so the page loop holds no broadcasts."""
        q4 = jnp.concatenate(
            [scr["iqv"][h_, pl.ds(t0, 4), :] for h_ in range(IH)], axis=0
        )  # rows 4h + q
        scr["b_qblk4"][...] = q4.astype(FP8).astype(scr["b_qblk4"].dtype)
        wq = scr["wts"][pl.ds(t0, 4), :]
        for h_ in range(IH):
            scr["b_wb"][h_] = jnp.broadcast_to(wq[:, h_ : h_ + 1], (4, pg))
        pos_q = rvec[pl.ds(t0, 4), 1:2]
        vis = (
            (seq - 1) if fault == "causal_mask" else jnp.minimum(pos_q, seq - 1)
        )  # FAULT: no causal limit
        lim = jnp.where(lax.broadcasted_iota(I32, (4, 1), 0) < nq, vis, -1)
        scr["b_lim"][...] = jnp.broadcast_to(lim, (4, pg))

    def scan(l):
        """Every request of the current wave: score its local pages into the key pool."""
        nblk = st[2]
        ring, ksem = scr["b_kring"], scr["b_kring_sem"]

        def run(ref):
            def issue_blk(i):
                b = scr["b_blk"][0, i]
                j = scr["b_blk"][1, i]
                npb = req[b, 3]
                slot = i % SD
                for p in range(NPG):

                    @pl.when(j * NPG + p < npb)
                    def _(p=p):
                        pj = jnp.minimum(j * NPG + p + PSH, MAXP - 1)
                        pltpu.make_async_copy(
                            ref.at[bti[b, pj]], ring.at[slot, p], ksem.at[slot]
                        ).start()

            for i0 in range(SD - 1):

                @pl.when(i0 < nblk)
                def _(i0=i0):
                    issue_blk(jnp.int32(i0))

            def blk_body(i, c):
                @pl.when(i + SD - 1 < nblk)
                def _():
                    issue_blk(i + SD - 1)

                b = scr["b_blk"][0, i]
                j = scr["b_blk"][1, i]
                slot = i % SD
                npb = req[b, 3]
                nlive = jnp.minimum(NPG, npb - j * NPG)

                def wbody(k, c2):
                    pltpu.make_async_copy(
                        ref.at[0], ring.at[slot, 0], ksem.at[slot]
                    ).wait()
                    return c2

                lax.fori_loop(0, nlive, wbody, 0)

                @pl.when(j == 0)
                def _():
                    scan_req_setup(req[b, 7], req[b, 8], req[b, 5])

                pbase = req[b, 4]
                lane_r4 = lax.broadcasted_iota(I32, (4, pg), 1)
                hb = req[b, 13] if VR > 1 else None

                def page(p):
                    blk = ring[slot, p]  # [IR4, 4, pg] u8
                    kT = pltpu.bitcast(blk[0 : IDIM // 4].reshape(IDIM, pg), FP8)
                    scale = (
                        pltpu.bitcast(
                            blk[IDIM // 4 : IDIM // 4 + 1].reshape(4, pg)[0:1],
                            jnp.float8_e8m0fnu,
                        )
                        .astype(BF16)
                        .astype(F32)
                    )  # [1, pg]
                    st_ = qdot(scr["b_qblk4"][...], kT)  # [4 * IH, pg]
                    acc = jnp.maximum(st_[0:4], 0.0) * scr["b_wb"][0]
                    for h_ in range(1, IH):  # production order
                        acc = (
                            acc
                            + jnp.maximum(st_[h_ * 4 : (h_ + 1) * 4], 0.0)
                            * scr["b_wb"][h_]
                        )
                    score = acc * scale
                    lidx = (j * NPG + p) * pg + lane_r4
                    if VR > 1:
                        gpos = gpos_of_l8(l8_of_tile(lidx, hb))
                    else:
                        gpos = (lidx // ILV) * G_ + me * ILV + lidx % ILV
                    ok = (gpos <= scr["b_lim"][...]) & (score != -jnp.inf)
                    key = jnp.where(ok, mono_key(score), jnp.int32(INT_MIN))  # [4, 256]
                    tile = jnp.concatenate([key[:, 0:128], key[:, 128:256]], axis=0)
                    scr["b_keys"][pbase + j * NPG + p] = tile

                PU = SCAN_UNROLL

                def pageu(
                    pp, c2
                ):  # PU independent pages per iteration (PU head-sum chains)
                    for u in range(PU):
                        page(PU * pp + u)
                    return c2

                lax.fori_loop(0, nlive // PU, pageu, 0)

                def page1(p, c2):
                    page(p)
                    return c2

                lax.fori_loop(nlive // PU * PU, nlive, page1, 0)
                return c

            lax.fori_loop(0, nblk, blk_body, 0)

        if "index" not in skip:
            csel(("idx",), l, run)

    def knew_prep():
        """The step's new index records (rec_v [TMAX, 256] bytes) as fp8 columns [128 dims, tokens] and
        their ue8m0 scales [1, tokens] (lanes past TMAX zero)."""
        rec = scr["rec_v"][...]  # [TMAX, 256] i32
        k8 = rec[:, 0:128]
        eb = rec[:, 128:129]
        if TMAX < 128:
            k8 = jnp.concatenate([k8, jnp.zeros((128 - TMAX, 128), I32)], axis=0)
            eb = jnp.concatenate([eb, jnp.zeros((128 - TMAX, 1), I32)], axis=0)
        kT = jnp.transpose(k8)  # [dims, tokens]
        scr["b_knew"][...] = lax.bitcast_convert_type(kT.astype(U8), FP8)
        e_row = jnp.transpose(jnp.broadcast_to(eb, (128, 128)))[
            0:8
        ]  # [8, tokens] ue8m0 bytes
        scr["b_snew"][...] = lax.bitcast_convert_type(
            shl(e_row, 23), F32
        )  # 2^(byte - 127) (byte 0: 0)

    def fix(bw0, bw1):
        """Patch the keys of this step's own new tokens (not in the HBM index cache yet) into the pool."""
        if fault in ("nofix", "kv_read_before_write"):
            return
        tok_l = lax.broadcasted_iota(I32, (8, 128), 1)

        def per_req(b, c):
            @pl.when(req[b, 10] > 0)
            def _():
                t0, nq, pbase = req[b, 7], req[b, 8], req[b, 4]
                build_qblk(t0)
                st_ = qdot(
                    scr["b_qblk"][...], scr["b_knew"][...]
                )  # [8 * IH, 128 tokens]
                score = head_sum(st_, t0) * scr["b_snew"][0:1, :]
                pos_q = dup4(rvec[pl.ds(t0, 4), 1:2])
                key = jnp.where(
                    score != -jnp.inf, mono_key(score), jnp.int32(INT_MIN)
                )  # [8, tokens]
                for qq in range(cfg.QMAX):
                    t2 = jnp.minimum(t0 + qq, TMAX - 1)
                    pos2 = rmeta[t2, 1]
                    lo2 = (pos2 // G_) * ILV + pos2 % ILV
                    if VR > 1:  # shard mk's position, on a page of this replica's parity
                        j8 = lo2 >> lg_pg
                        hit = jnp.logical_and(
                            jnp.logical_and(qq < nq, (pos2 % G_) // ILV == mk),
                            (j8 + VR - req[b, 13]) % VR == 0,
                        )
                    else:
                        hit = jnp.logical_and(qq < nq, (pos2 % G_) // ILV == me)

                    @pl.when(hit)
                    def _(t2=t2, lo2=lo2, pos2=pos2):
                        kc = pltpu.roll(key, (128 - t2) % 128, 1)[
                            :, 0:1
                        ]  # [8, 1] token t2
                        kc = jnp.where(sub8[:, 0:1] < 4, kc, pltpu.roll(kc, 4, 0))
                        caus = (
                            (pos_q[:, 0:1] >= pos2)
                            if fault != "causal_mask"
                            else (pos_q[:, 0:1] >= 0)
                        )
                        kc = jnp.where(
                            caus & (sub8[:, 0:1] % 4 < nq), kc, jnp.int32(INT_MIN)
                        )
                        ti = pbase + ((lo2 >> lg_pg) // VR if VR > 1 else (lo2 >> lg_pg))
                        cc = (lo2 >> 7) & 1
                        old = scr["b_keys"][ti]
                        scr["b_keys"][ti] = jnp.where(
                            (sub8 // 4 == cc) & (tok_l == (lo2 & 127)), kc, old
                        )

            return c

        lax.fori_loop(bw0, bw1, per_req, 0)

    def grp_ranges(bw0, bw1):
        """Per 8-row group: the requests of the wave whose rows intersect it -> b_grp[0/1, g]."""
        for gi in range(NG):
            scr["b_grp"][0, gi] = bw1
            scr["b_grp"][1, gi] = bw0

        def body(b, c):
            t0, nq = req[b, 7], req[b, 8]
            for gi in range(NG):
                hit = jnp.logical_and(
                    nq > 0, jnp.logical_and(t0 < 8 * gi + 8, t0 + nq > 8 * gi)
                )

                @pl.when(hit)
                def _(gi=gi):
                    scr["b_grp"][0, gi] = jnp.minimum(scr["b_grp"][0, gi], b)
                    scr["b_grp"][1, gi] = jnp.maximum(scr["b_grp"][1, gi], b + 1)

            return c

        lax.fori_loop(bw0, bw1, body, 0)

    ones_bf = jnp.ones((128, 128), BF16)
    cur_l = [None]  # the layer of the running indexer_tail (fault controls)
    FIRST_FRESH = min([l_ for l_ in range(cfg.L) if cfg.full_indexer(l_)] or [0])

    def lane_sum_small(a):
        """(n, 128) i32 with entries <= 256 -> per-row lane sums broadcast to all lanes (exact: bf16 holds
        integers <= 256, f32 accumulation), on the MXU instead of XLU reduces + converts."""
        return lax.dot_general(
            a.astype(BF16), ones_bf, NN, preferred_element_type=F32
        ).astype(I32)

    def thr_parts(DW, rt0, rt1):
        """The radix threshold's per-group pieces for digit width DW over the rows [rt0, rt1) of the current wave:
        count_payload (count, no send), count_send (stage A: count + send group gi), finish (stage A: wait group gi's
        exchange + decide) and finish_from (L1-M: decide group gi from already-received payloads, lane offset off)."""
        NK = (1 << DW) - 1
        KMAX = NK + 1  # last round: k = 0 .. 2^DW

        def parity(gi, r):
            """Stage A: r % 2. L1-E: a group may stop after any round, so its parity continues across calls
            (base + r) % 2 -- the next call's first send never lands in a buffer a peer may still be reading."""
            return (scr["b_tg"][1, gi] + r) % 2 if EARLY else r % 2

        def count_payload(gi, r, last, off=None, payload0=None):
            """Group gi's round-r counts: row (sublane) x lane k (+ off) = #(key >= splitter k) on this rank.
            Returns (payload, parity) -- the parity computed where stage A computed it (identical program)."""
            sh = 32 - DW * (r + 1)
            par = parity(gi, r)
            ks = list(range(0, KMAX + 1)) if last else list(range(1, NK + 1))

            def swar_tots(pref, en, sh, pbase, npb, ks):
                """L1-S (DW = 4): per-row counts #(key >= pref + k << sh) for k in ks, from a 16-bin histogram of the
                in-bracket digit (four i32 words of 8-bit counters, flushed every SW_CH tiles) plus the above-bracket
                count: count_k = above + sum_{j >= k} bin_j. Exactly stage A's compare counts, en / wrap included."""
                cpref = jnp.where(en, pref ^ INT_MIN, INT_MAX)
                cu = pref + shl(jnp.int32(16), sh)
                # bracket end past 2^32: an unsigned overflow of pref + 16 << sh, or 16 << sh itself >= 2^32 (sh >= 28,
                # the first round), which int32 wraps to 0 without tripping the overflow test (T2 j-104eabdd: RED)
                wrap = ((cu ^ INT_MIN) < (pref ^ INT_MIN)) | (sh >= 32 - 4)
                cend = jnp.where(en & ~wrap, cu ^ INT_MIN, INT_MAX)
                one = jnp.ones((8, 128), I32)
                z = jnp.zeros((8, 128), I32)

                def tile_parts(t_):
                    ab = t_ >= cend
                    inb = (t_ >= cpref) & ~ab
                    d = shr(t_ - cpref, sh)  # in-bracket digit in [0, 16)
                    inc = jnp.where(inb, shl(one, shl(d & 3, 3)), 0)
                    wi = shr(d, 2)
                    return [jnp.where(wi == w_, inc, 0) for w_ in range(4)] + [
                        ab.astype(I32)
                    ]

                def tsum(xs):
                    while len(xs) > 1:
                        xs = [xs[j] + xs[j + 1] for j in range(0, len(xs) - 1, 2)] + (
                            [xs[-1]] if len(xs) % 2 else []
                        )
                    return xs[0]

                def chunk(c, carry):
                    bins, abt = list(carry[:16]), carry[16]
                    lo = c * SW_CH
                    hi = jnp.minimum(npb, lo + SW_CH)
                    nfull = (hi - lo) // THR_U

                    def tbu(i, st):
                        base = pbase + lo + i * THR_U
                        parts = [
                            tile_parts(scr["b_keys"][base + u]) for u in range(THR_U)
                        ]
                        return tuple(
                            st[j] + tsum([pp[j] for pp in parts]) for j in range(5)
                        )

                    def tb1(i, st):
                        parts = tile_parts(scr["b_keys"][pbase + i])
                        return tuple(st[j] + parts[j] for j in range(5))

                    st = lax.fori_loop(0, nfull, tbu, (z, z, z, z, z))
                    st = lax.fori_loop(lo + nfull * THR_U, hi, tb1, st)
                    for j in range(16):
                        bins[j] = bins[j] + (shr(st[j >> 2], 8 * (j & 3)) & 0xFF)
                    return tuple(bins) + (abt + st[4],)

                nch = (npb + SW_CH - 1) // SW_CH
                fin = lax.fori_loop(0, nch, chunk, tuple(z for _ in range(17)))
                if MAXP_OK:
                    sums = lane_sum_small(
                        jnp.concatenate(list(fin), axis=0)
                    )  # [8 * 17, 128]
                    bt = [sums[8 * i : 8 * i + 8, 0:1] for i in range(17)]
                else:
                    bt = [jnp.sum(a, axis=1, keepdims=True) for a in fin]
                S = [None] * 17
                S[16] = bt[16]
                for j in range(15, -1, -1):
                    S[j] = S[j + 1] + bt[j]
                return [S[k] for k in ks]

            def per_req(b, payload):
                t0, nq, pbase, npb = req[b, 7], req[b, 8], req[b, 4], req[b, 3]
                pref = rows4(1, t0)
                mode = rows4(0, t0)
                trow = t0 + (sub8 % 4)
                en = (
                    (mode == 2)
                    & (sub8 % 4 < nq)
                    & (trow >= 8 * gi)
                    & (trow < 8 * gi + 8)
                )
                if (
                    THR_SWAR and DW == 4
                ):  # L1-S: digit histogram in packed 8-bit counters (exact)
                    tots = swar_tots(pref, en, sh, pbase, npb, ks)
                    return place(tots, ks, t0, nq, payload)
                cs = []
                for k in ks:
                    cu = pref + shl(jnp.int32(k), sh) if k else pref
                    if last and k == KMAX:
                        wrap = (cu ^ INT_MIN) < (pref ^ INT_MIN)  # unsigned overflow
                        cs.append(jnp.where(en & ~wrap, cu ^ INT_MIN, INT_MAX))
                    else:
                        cs.append(jnp.where(en, cu ^ INT_MIN, INT_MAX))

                def tbody(i, accs):
                    tile = scr["b_keys"][pbase + i]
                    return tuple(
                        a + (tile >= c_).astype(I32) for a, c_ in zip(accs, cs)
                    )

                z = jnp.zeros((8, 128), I32)
                if THR_U == 1:  # stage A: one tile per iteration
                    accs = lax.fori_loop(0, npb, tbody, tuple(z for _ in ks))
                else:  # L1-U: THR_U tiles per iteration, per-splitter tree sum (exact)

                    def tbodyu(i, accs):
                        base = pbase + i * THR_U
                        tl = [scr["b_keys"][base + u] for u in range(THR_U)]
                        outs = []
                        for a, c_ in zip(accs, cs):
                            ms = [(t_ >= c_).astype(I32) for t_ in tl]
                            while len(ms) > 1:
                                ms = [
                                    ms[j] + ms[j + 1] for j in range(0, len(ms) - 1, 2)
                                ] + ([ms[-1]] if len(ms) % 2 else [])
                            outs.append(a + ms[0])
                        return tuple(outs)

                    nfull = npb // THR_U
                    accs = lax.fori_loop(0, nfull, tbodyu, tuple(z for _ in ks))
                    accs = lax.fori_loop(nfull * THR_U, npb, tbody, accs)
                mb = jnp.zeros((8, 128), I32)
                if MAXP_OK:  # all lane sums in one MXU pass (weights pushed once)
                    sums = lane_sum_small(
                        jnp.concatenate(list(accs), axis=0)
                    )  # [8 * len(ks), 128]
                    tots = [sums[8 * i : 8 * i + 8, 0:1] for i in range(len(ks))]
                else:
                    tots = [jnp.sum(a, axis=1, keepdims=True) for a in accs]
                return place(tots, ks, t0, nq, payload, mb)

            def place(tots, ks, t0, nq, payload, mb=None):
                if mb is None:
                    mb = jnp.zeros((8, 128), I32)
                for k, tot in zip(ks, tots):
                    tot = tot + pltpu.roll(tot, 4, 0)
                    mb = jnp.where(lane8 == (k if off is None else k + off), tot, mb)
                o = t0 - 8 * gi
                rolled = pltpu.roll(mb, (o + 8) % 8, 0)
                m = (sub8 >= o) & (sub8 < o + jnp.minimum(nq, 4))
                if (
                    off is not None
                ):  # merged payload: leave the other groups' lane slots alone
                    m = m & (lane8 >= off) & (lane8 < off + MSLOT)
                return jnp.where(m, rolled, payload)

            return lax.fori_loop(
                scr["b_grp"][0, gi],
                scr["b_grp"][1, gi],
                per_req,
                jnp.zeros((8, 128), I32) if payload0 is None else payload0,
            ), par

        def count_send(gi, r, last):
            payload, par = count_payload(gi, r, last)
            if DSEND:  # L1-D: b_csend[gi, par] was last sent in round r - 2
                if isinstance(r, int):
                    if r >= 2:
                        cx_wait_send(gi, par)
                else:

                    @pl.when(r >= 2)
                    def _():
                        cx_wait_send(gi, par)

            scr["b_csend"][gi, par] = payload
            scr["b_crecv"][gi, par, me] = payload
            if getattr(E, "delay", None) is not None:
                E.delay(gi, r)
            cx_start(gi, par)

        def decide(gi, sh, last, crecv, Gs, off, r=None):
            """Digit (and on the last round T / P / allowance) of group gi's rows from the summed counts Gs and the
            per-rank payloads crecv, whose lanes [off, off + K) are this group's (sh: the round's digit shift)."""
            lo_, hi_ = (1, NK) if off is None else (1 + off, NK + off)
            ge = (Gs >= KT) & (lane8 >= lo_) & (lane8 <= hi_)
            d = jnp.sum(ge.astype(I32), axis=1, keepdims=True)  # (8, 1) digit
            if fault == "digit" or (fault == "f_thr2" and DW == 2):
                d = jnp.minimum(d + 1, NK)
            if fault == "f_wave":
                d = jnp.where(rt0 > 0, jnp.minimum(d + 1, NK), d)
            r8 = pl.multiple_of(gi * 8, 8)
            pref = row[1, pl.ds(r8, 8), :]
            if not last:
                newp = pref | shl(jnp.broadcast_to(d, (8, 128)), sh)
                row[1, pl.ds(r8, 8), :] = newp
                if EARLY:  # L1-E: exact early stop of this group
                    # #(key >= new prefix) is the summed count at splitter d (d >= 1), else unchanged
                    gsel = jnp.sum(jnp.where(lane8 == d, Gs, 0), axis=1, keepdims=True)
                    nlo = jnp.where(d >= 1, gsel, row[5, pl.ds(r8, 8), 0:1])
                    row[5, pl.ds(r8, 8), :] = jnp.broadcast_to(nlo, (8, 128))
                    mode = row[0, pl.ds(r8, 8), 0:1]
                    trow = gi * 8 + lax.broadcasted_iota(I32, (8, 1), 0)
                    inw = (trow >= rt0) & (trow < rt1)
                    rad = (mode == 2) & inw
                    allok = jnp.min(jnp.where(rad, jnp.where(nlo == KT, 1, 0), 1))

                    # every radix row of the group has exactly KT keys >= its prefix: {key >= prefix} IS the
                    # top-KT set (no key lies between the prefix and the true T, and every key == T is selected),
                    # so T = prefix with P = all ranks' ties selects the same set as the full radix.
                    @pl.when(allok == 1)
                    def _():
                        T_f = jnp.where(
                            mode == 2,
                            newp[:, 0:1] ^ INT_MIN,
                            jnp.where(mode == 1, INT_MIN, INT_MAX),
                        )
                        P_f = jnp.where(mode == 2, INT_MAX, 0)
                        for k_, v_ in ((2, T_f), (3, P_f), (4, jnp.zeros((8, 1), I32))):
                            old = row[k_, pl.ds(r8, 8), :]
                            row[k_, pl.ds(r8, 8), :] = jnp.where(
                                inw, jnp.broadcast_to(v_, (8, 128)), old
                            )
                        scr["b_tg"][0, gi] = jnp.int32(1)
                        if DSEND:  # this group's pending sends: rounds r - 1 and r

                            @pl.when(r >= 1)
                            def _():
                                cx_wait_send(gi, parity(gi, r - 1))

                            cx_wait_send(gi, parity(gi, r))
                        scr["b_tg"][1, gi] = scr["b_tg"][1, gi] + r + 1

                return
            Tu = pref[:, 0:1] | d
            Ts = Tu ^ INT_MIN
            dl = d if off is None else d + off
            ge_s, gt_s = [], []
            for s_ in range(R):
                P_ = crecv[s_]
                ge_s.append(
                    jnp.sum(jnp.where(lane8 == dl, P_, 0), axis=1, keepdims=True)
                )
                gt_s.append(
                    jnp.sum(jnp.where(lane8 == dl + 1, P_, 0), axis=1, keepdims=True)
                )
            n_gt = gt_s[0]
            for x in gt_s[1:]:
                n_gt = n_gt + x
            need = KT - n_gt
            eqs = [a - b for a, b in zip(ge_s, gt_s)]
            before = jnp.zeros((8, 1), I32)
            eq_me = jnp.zeros((8, 1), I32)
            vme = VR * (me % D_) + me // D_ if VR > 1 else None
            for s_ in range(R):
                if fault == "tierank":
                    before = before + jnp.where(s_ > me, eqs[s_], 0)
                elif VR > 1 and fault != "f_vorder":  # V16: virtual rank order v(s) = VR * (s % D) + s // D
                    before = before + jnp.where(VR * (s_ % D_) + s_ // D_ < vme, eqs[s_], 0)
                else:
                    before = before + jnp.where(s_ < me, eqs[s_], 0)
                eq_me = jnp.where(s_ == me, eqs[s_], eq_me)
            a = jnp.clip(need - before, 0, eq_me)
            mode = row[0, pl.ds(r8, 8), 0:1]
            T_f = jnp.where(mode == 2, Ts, jnp.where(mode == 1, INT_MIN, INT_MAX))
            P_f = jnp.where(
                mode == 2, jnp.where(a == eq_me, INT_MAX, jnp.where(a == 0, 0, -1)), 0
            )
            trow = gi * 8 + lax.broadcasted_iota(I32, (8, 1), 0)
            inw = (trow >= rt0) & (trow < rt1)
            for k_, v_ in ((2, T_f), (3, P_f), (4, a)):
                old = row[k_, pl.ds(r8, 8), :]
                row[k_, pl.ds(r8, 8), :] = jnp.where(
                    inw, jnp.broadcast_to(v_, (8, 128)), old
                )

        def finish(gi, r, last):
            sh = 32 - DW * (r + 1)
            par = parity(gi, r)
            early = (
                fault == "f_cx_earlyread"
            )  # FAULT: read the peers' payloads before waiting their receives
            if early:
                crecv = [scr["b_crecv"][gi, par, s_] for s_ in range(R)]
            cx_finish(gi, par)
            if (
                "coll" in skip
            ):  # removal arm (timing only): pretend every rank counted like me
                for s_ in range(R):

                    @pl.when(me != s_)
                    def _(s_=s_):
                        scr["b_crecv"][gi, par, s_] = scr["b_crecv"][gi, par, me]

            if not early:
                crecv = [scr["b_crecv"][gi, par, s_] for s_ in range(R)]
            Gs = crecv[0]
            for s_ in range(1, R):
                Gs = Gs + crecv[s_]
            decide(gi, sh, last, crecv, Gs, None, r=r)

        return dict(
            NR=32 // DW,
            NK=NK,
            KMAX=KMAX,
            count_payload=count_payload,
            count_send=count_send,
            finish=finish,
            decide=decide,
            parity=parity,
        )

    def threshold(DW, rt0, rt1, g0, g1, after_group):
        """Exact distributed radix threshold for the radix rows of groups [g0, g1) restricted to rows
        [rt0, rt1) (the current wave); writes per-row T / P / allowance into b_row (rows of the wave).
        after_group(g): called once group g's rows are final (the union of requests ending in g)."""
        cp = thr_parts(DW, rt0, rt1)
        NR, count_send, finish = cp["NR"], cp["count_send"], cp["finish"]
        if EARLY:
            return threshold_early(cp, rt0, rt1, g0, g1, after_group)

        def init(gi, c):
            r8 = pl.multiple_of(gi * 8, 8)
            row[1, pl.ds(r8, 8), :] = jnp.zeros((8, 128), I32)
            return c

        lax.fori_loop(g0, g1, init, 0)

        def rbody(r, c):
            def gbody(gi, c2):
                @pl.when(r > 0)
                def _():
                    finish(gi, r - 1, False)

                count_send(gi, r, False)
                return c2

            lax.fori_loop(g0, g1, gbody, 0)
            return c

        lax.fori_loop(0, NR - 1, rbody, 0)

        def glast(gi, c):
            if NR > 1:
                finish(gi, NR - 2, False)
            count_send(gi, NR - 1, True)
            return c

        lax.fori_loop(g0, g1, glast, 0)

        def gfin(gi, c):
            finish(gi, NR - 1, True)
            if DSEND:  # L1-D: drain the last two rounds' sends of this group
                cx_wait_send(gi, (NR - 2) % 2)
                cx_wait_send(gi, (NR - 1) % 2)
            after_group(gi)  # overlaps the union with later groups' exchanges
            return c

        lax.fori_loop(g0, g1, gfin, 0)

    def threshold_early(cp, rt0, rt1, g0, g1, after_group):
        """L1-E: threshold() whose groups stop independently once every radix row has exactly KT keys >= its prefix
        (decide()). A stopped group skips its remaining sends and decides; all ranks stop it in the same round (the
        decision is on the summed counts). Parity bases persist across calls (parity())."""
        NR, count_send, finish = cp["NR"], cp["count_send"], cp["finish"]
        tg = scr["b_tg"]

        def init(gi, c):
            r8 = pl.multiple_of(gi * 8, 8)
            row[1, pl.ds(r8, 8), :] = jnp.zeros((8, 128), I32)
            row[5, pl.ds(r8, 8), :] = jnp.full((8, 128), INT_MAX, I32)
            tg[0, gi] = jnp.int32(0)
            return c

        lax.fori_loop(g0, g1, init, 0)

        def rbody(r, c):
            def gbody(gi, c2):
                @pl.when(jnp.logical_and(r > 0, tg[0, gi] == 0))
                def _():
                    finish(gi, r - 1, False)

                @pl.when(tg[0, gi] == 0)
                def _():
                    count_send(gi, r, False)

                return c2

            lax.fori_loop(g0, g1, gbody, 0)
            return c

        lax.fori_loop(0, NR - 1, rbody, 0)

        def glast(gi, c):
            @pl.when(tg[0, gi] == 0)
            def _():
                finish(gi, NR - 2, False)

            @pl.when(tg[0, gi] == 0)
            def _():
                count_send(gi, NR - 1, True)

            return c

        lax.fori_loop(g0, g1, glast, 0)

        def gfin(gi, c):
            @pl.when(tg[0, gi] == 0)
            def _():
                finish(gi, NR - 1, True)
                if DSEND:
                    cx_wait_send(gi, cp_parity(gi, NR - 2))
                    cx_wait_send(gi, cp_parity(gi, NR - 1))
                tg[1, gi] = tg[1, gi] + NR

            after_group(gi)
            return c

        cp_parity = cp["parity"]
        lax.fori_loop(g0, g1, gfin, 0)

    def threshold_merged(DW, rt0, rt1, g0, g1, after_group):
        """L1-M: threshold() with ONE count exchange per round. Every group's counts travel in one (8, 128) vreg,
        group gi in lanes [MSLOT * gi, MSLOT * gi + K): a round sends 15 x 4 KiB and waits 2 x 15 semaphores,
        instead of NG times that. Same counts, same digits, same T / P / allowance (exact)."""
        cp = thr_parts(DW, rt0, rt1)
        NR, KMAX = cp["NR"], cp["KMAX"]
        assert NG * MSLOT <= 128 and KMAX + 1 <= MSLOT, (NG, MSLOT, DW)

        def init(gi, c):
            r8 = pl.multiple_of(gi * 8, 8)
            row[1, pl.ds(r8, 8), :] = jnp.zeros((8, 128), I32)
            return c

        lax.fori_loop(g0, g1, init, 0)

        def round_(r, last):
            sh = 32 - DW * (r + 1)
            par = r % 2

            def gcount(gi, merged):
                return cp["count_payload"](
                    gi, r, last, off=MSLOT * gi, payload0=merged
                )[0]

            merged = lax.fori_loop(g0, g1, gcount, jnp.zeros((8, 128), I32))
            scr["b_csend"][0, par] = merged
            scr["b_crecv"][0, par, me] = merged
            if getattr(E, "delay", None) is not None:
                E.delay(0, r)
            cx_start(0, par)
            early = (
                fault == "f_cx_earlyread"
            )  # FAULT: read the peers' payloads before waiting their receives
            if early:
                crecv = [scr["b_crecv"][0, par, s_] for s_ in range(R)]
            cx_finish(0, par)
            if (
                "coll" in skip
            ):  # removal arm (timing only): pretend every rank counted like me
                for s_ in range(R):

                    @pl.when(me != s_)
                    def _(s_=s_):
                        scr["b_crecv"][0, par, s_] = scr["b_crecv"][0, par, me]

            if not early:
                crecv = [scr["b_crecv"][0, par, s_] for s_ in range(R)]
            Gs = crecv[0]
            for s_ in range(1, R):
                Gs = Gs + crecv[s_]

            def gfin(gi, c):
                cp["decide"](gi, sh, last, crecv, Gs, MSLOT * gi)
                return c

            lax.fori_loop(g0, g1, gfin, 0)
            if last:  # unions after every group's rows are final (no exchange left)

                def gu(gi, c):
                    after_group(gi)
                    return c

                lax.fori_loop(g0, g1, gu, 0)

        def rbody(r, c):
            round_(r, False)
            return c

        lax.fori_loop(0, NR - 1, rbody, 0)
        round_(NR - 1, True)

    def set_short(rt0, rt1):
        """No radix row in the wave: T / P straight from the modes (rows of the wave only)."""
        mode = row[0]
        T_f = jnp.where(mode == 1, INT_MIN, INT_MAX)
        tr = lax.broadcasted_iota(I32, (TR, 128), 0)
        inw = (tr >= rt0) & (tr < rt1)
        row[2] = jnp.where(inw, T_f, row[2])
        row[3] = jnp.where(inw, 0, row[3])

    def tie_bound(b):
        """Rows of request b whose share of the tied keys is partial (P == -1): the position bound P with
        #(key == T, pos < P) == allowance, by a PB-bit search over the request's tiles."""
        t0, nq, pbase, npb = req[b, 7], req[b, 8], req[b, 4], req[b, 3]
        Tc = rows4(2, t0)
        Pc = rows4(3, t0)
        ac = rows4(4, t0)
        need_tb = (Pc == -1) & (sub8 % 4 < nq)

        def bit(i, cand):
            trial = cand | shl(jnp.ones((8, 128), I32), PB - 1 - i)

            def tb(p, acc):
                tile = scr["b_keys"][pbase + p]
                pos = p * pg + 128 * (sub8 // 4) + lane8
                return acc + ((tile == Tc) & (pos < trial)).astype(I32)

            acc = lax.fori_loop(0, npb, tb, jnp.zeros((8, 128), I32))
            tot = jnp.sum(acc, axis=1, keepdims=True)
            tot = tot + pltpu.roll(tot, 4, 0)
            return jnp.where(tot <= ac, trial, cand)

        cand = lax.fori_loop(0, PB, bit, jnp.zeros((8, 128), I32))
        newP = jnp.where(need_tb, cand, Pc)
        for q in range(4):

            @pl.when(q < nq)
            def _(q=q):
                row[3, pl.ds(t0 + q, 1), :] = newP[q : q + 1]

    def union_req(b):
        """Request b: tie bounds, membership bits -> extraction -> own_uv rows at st[4]; then each entry is
        resolved in place to its gather source (page << 8 | slot, or staging row << 2 | seg for this step's
        own tokens; the low 2 bits stay the rope quarter) so every layer issues from the same table."""
        t0, nq, pbase, npb = req[b, 7], req[b, 8], req[b, 4], req[b, 3]
        r0 = st[4]
        ok = jnp.logical_and(req[b, 10] > 0, npb > 0)
        req[b, 0] = jnp.int32(0)
        req[b, 1] = r0
        req[b, 2] = jnp.int32(0)

        @pl.when(ok)
        def _():
            Pq = row[3, pl.ds(t0, 4), :]
            need = (Pq == -1) & (lax.broadcasted_iota(I32, (4, 128), 0) < nq)

            @pl.when(jnp.max(jnp.where(need, 1, 0)) > 0)
            def _():
                tie_bound(b)

            Tc = rows4(2, t0)
            Pc = rows4(3, t0)
            live = sub8 % 4 < nq
            Tc = jnp.where(live, Tc, INT_MAX)
            Pc = jnp.where(live, Pc, 0)
            bitv = shl(jnp.ones((8, 128), I32), sub8 % 4)
            nold = req[b, 6]
            lane1 = lax.broadcasted_iota(I32, (1, 128), 1)
            swap = None
            if fault == "topk_swap":
                # FAULT (glm_ref topk_swap, first fresh layer): on rank 0, request 0's row 0 drops its first
                # selected entry and takes its lowest-scoring unselected visible position instead
                r0m = sub8 % 4 == 0
                BIGP = jnp.int32(1 << 30)

                def selm(p):
                    tile = scr["b_keys"][pbase + p]
                    pos = p * pg + 128 * (sub8 // 4) + lane8
                    return tile, pos, (tile > Tc) | ((tile == Tc) & (pos < Pc))

                def pass1(p, c):
                    pf, km = c
                    tile, pos, sel = selm(p)
                    pf = jnp.minimum(pf, jnp.min(jnp.where(r0m & sel, pos, BIGP)))
                    km = jnp.minimum(
                        km,
                        jnp.min(
                            jnp.where(r0m & ~sel & (tile > INT_MIN), tile, INT_MAX)
                        ),
                    )
                    return pf, km

                pfirst, kmin = lax.fori_loop(0, npb, pass1, (BIGP, jnp.int32(INT_MAX)))

                def pass2(p, c):
                    tile, pos, sel = selm(p)
                    return jnp.minimum(
                        c,
                        jnp.min(
                            jnp.where(
                                r0m & ~sel & (tile == kmin) & (tile > INT_MIN),
                                pos,
                                BIGP,
                            )
                        ),
                    )

                pmin = lax.fori_loop(0, npb, pass2, BIGP)
                tgt = (
                    (cur_l[0] == FIRST_FRESH)
                    & (b == 0)
                    & (me == 0)
                    & (pfirst < BIGP)
                    & (pmin < BIGP)
                )
                swap = (r0m, jnp.where(tgt, pfirst, -1), jnp.where(tgt, pmin, -1))

            def mb(p4, cnt):
                rws = []
                for u in range(4):
                    p = p4 * 4 + u
                    tile = scr["b_keys"][pbase + jnp.minimum(p, npb - 1)]
                    pos = p * pg + 128 * (sub8 // 4) + lane8
                    sel = ((tile > Tc) | ((tile == Tc) & (pos < Pc))) & (p < npb)
                    if swap is not None:
                        r0m_, drop_, add_ = swap
                        sel = (sel & ~(r0m_ & (pos == drop_))) | (
                            r0m_ & (pos == add_) & (p < npb)
                        )
                    bits = jnp.where(sel, bitv, 0)
                    for c_ in range(2):
                        rr = jnp.sum(bits[4 * c_ : 4 * c_ + 4], axis=0, keepdims=True)
                        rws.append(rr)
                        cnt = cnt + jnp.where(
                            (rr > 0) & (p * pg + 128 * c_ + lane1 < nold), 1, 0
                        )
                scr["b_memb"][pl.ds(pl.multiple_of(p4 * 8, 8), 8), :] = jnp.concatenate(
                    rws, axis=0
                )
                return cnt

            n4 = (npb + 3) // 4
            cnt = lax.fori_loop(0, n4, mb, jnp.zeros((1, 128), I32))
            nuo = jnp.sum(cnt)
            nbk = (2 * npb + 127) // 128

            def zr(i, c):
                scr["b_memb"][pl.ds(pl.multiple_of(i * 8, 8), 8), :] = jnp.zeros(
                    (8, 128), I32
                )
                return c

            lax.fori_loop(n4, nbk * 16, zr, 0)
            n = extract_union(
                scr["b_memb"],
                nbk,
                scr["own_uv"],
                r0,
                scr["b_pre"],
                scr["b_xacc"],
                scr["b_xcc"],
                XKC,
                MB_SH,
                max_rows=UR2 - r0,
            )
            nrow = (n + 127) // 128
            p0 = req[b, 9]
            BIG = jnp.int32(1 << 30)

            def rrow(
                i, c
            ):  # resolve in place, vector (pages: compare-select over the row's range)
                e = scr["own_uv"][pl.ds(r0 + i, 1), :]
                m = e & SRC_MASK
                kk = i * 128 + lane1
                lv = kk < n
                if (
                    getattr(E, "dbg_uv", None) is not None
                ):  # debug: the pre-resolution entry (idx | bits << MB_SH)
                    E.dbg_uv[pl.ds(r0 + i, 1), :] = e
                old = (
                    (lv & (m < nold)) if fault != "kv_read_before_write" else lv
                )  # FAULT: new rows read stale
                pidx = jnp.where(old, m >> lg_pg, -1)
                lo = jnp.min(jnp.where(old, pidx, BIG))
                hi = jnp.max(pidx)

                def pgb(jj, pv):
                    return jnp.where(
                        pidx == jj, bt[b, jnp.minimum(jj + PSH, MAXP - 1)], pv
                    )

                page = lax.fori_loop(
                    jnp.minimum(lo, hi + 1), hi + 1, pgb, jnp.zeros((1, 128), I32)
                )
                if fault == "page":
                    page = jnp.where(
                        page + 1 < E.nb, page + 1, 0
                    )  # stays in bounds (no UB on device)
                if VR > 1:
                    gpos = gpos_of_l8(l8_of_tile(m, req[b, 13]))
                else:
                    gpos = (m // ILV) * G_ + me * ILV + m % ILV
                t2 = jnp.clip(t0 + (gpos - p0), 0, TMAX - 1)
                src = jnp.where(
                    old, shl(page, 8) | (m & (pg - 1)), shl(t2, 2) | (m & 3)
                )
                scr["own_uv"][pl.ds(r0 + i, 1), :] = jnp.where(
                    lv, src | (e & ~SRC_MASK), e
                )
                scr["b_greq"][r0 + i] = b
                return c

            lax.fori_loop(0, nrow, rrow, 0)
            req[b, 0] = n
            req[b, 2] = (
                n if fault == "kv_read_before_write" else nuo
            )  # FAULT: every entry resolved as an old row
            st[4] = r0 + nrow

    def union_upto(bw0, bw1, limit_row):
        """Unions of the wave's requests st[5].. whose last row is < limit_row (their rows are final); requests
        are ordered by first row, so this is a prefix (a fori + when: Mosaic rejected the while form)."""

        def body(b, c):
            go = jnp.logical_and(
                b >= st[5], req[b, 7] + jnp.maximum(req[b, 8], 1) - 1 < limit_row
            )

            @pl.when(go)
            def _():
                if tiles.trace:
                    with jax.named_scope("I_unionT"):
                        union_req(b)
                else:
                    union_req(b)
                st[5] = b + 1

            return c

        lax.fori_loop(bw0, bw1, body, 0)

    def udma(U):
        """own_uv rows [0, U) -> own_u (SMEM), in power-of-two row chunks. UV_ALIAS_FAULTS builds: rows
        [0, ceil8(U)) in chunks of >= 8 rows at 8-row-aligned offsets (own_uv is an aliased tiled view there)."""
        sem = E.misc_sem
        cps = []
        uv_alias = fault in UV_ALIAS_FAULTS
        nmax = uv_rows(g, tiles)
        if uv_alias:
            U = (U + 7) & ~7  # <= nmax: U <= UR2 - XKC / 128, so ceil8(U) <= ceil8(UR2)
        for k in range(int(math.ceil(math.log2(nmax + 1))), -1, -1):
            n_ = 1 << k
            if n_ > nmax or (uv_alias and n_ < 8):
                continue
            off = U & ~(2 * n_ - 1)
            if uv_alias:
                off = pl.multiple_of(off, 8)  # a multiple of 2 * n_ >= 16
            cps.append((n_, off))
        for n_, off in cps:

            @pl.when((U & n_) != 0)
            def _(n_=n_, off=off):
                pltpu.make_async_copy(
                    scr["own_uv"].at[pl.ds(off, n_)],
                    scr["own_u"].at[pl.ds(off, n_)],
                    sem,
                ).start()

        for n_, off in cps:

            @pl.when((U & n_) != 0)
            def _(n_=n_, off=off):
                pltpu.make_async_copy(
                    scr["own_uv"].at[pl.ds(off, n_)],
                    scr["own_u"].at[pl.ds(off, n_)],
                    sem,
                ).wait()

    def indexer_tail(l):
        """After I_proj / record: the batched index path for every wave -> union rows (own_uv, own_u)."""
        cur_l[0] = l
        pi = Phases(tiles.trace)
        pi.mark("I_fixp")
        knew_prep()
        nw = st[1]

        def wave(w, row0):
            bw0 = scr["b_wave"][w]
            bw1 = scr["b_wave"][w + 1]
            pw = Phases(tiles.trace)
            pw.mark("I_scan")

            # pool bases and the scan block list of the wave
            def pb(b, c):
                acc, nb_, tm = c
                req[b, 4] = acc
                npb = req[b, 3]
                nbb = (npb + NPG - 1) // NPG

                def bl(j, c2):
                    scr["b_blk"][0, nb_ + j] = b
                    scr["b_blk"][1, nb_ + j] = j
                    return c2

                lax.fori_loop(0, nbb, bl, 0)
                return acc + npb, nb_ + nbb, tm + req[b, 11]

            # tot: rank-0 page count of the wave (replicated: every rank picks the same digit width)
            _, nblk, tot = lax.fori_loop(
                bw0, bw1, pb, (jnp.int32(0), jnp.int32(0), jnp.int32(0))
            )
            st[2] = nblk
            scan(l)
            pw.mark("I_fix")
            fix(bw0, bw1)
            pw.mark("I_thr")
            T = T_()
            rt0 = req[jnp.minimum(bw0, BMAX - 1), 7]
            rt1 = jnp.where(bw1 >= B_(), T, req[jnp.minimum(bw1, BMAX - 1), 7])
            mode = row[0]
            tr = lax.broadcasted_iota(I32, (TR, 128), 0)
            anyr = jnp.max(jnp.where((mode == 2) & (tr >= rt0) & (tr < rt1), 1, 0))
            g0 = rt0 // 8
            g1 = (rt1 + 7) // 8
            grp_ranges(bw0, bw1)

            st[5] = bw0

            def after(gi):
                return union_upto(bw0, bw1, 8 * (gi + 1))

            def thr(DW):
                if MERGE == "never":
                    threshold(DW, rt0, rt1, g0, g1, after)
                elif MERGE == "always":
                    threshold_merged(DW, rt0, rt1, g0, g1, after)
                else:

                    @pl.when(tot <= tiles.thr_merge)
                    def _():
                        threshold_merged(DW, rt0, rt1, g0, g1, after)

                    @pl.when(tot > tiles.thr_merge)
                    def _():
                        threshold(DW, rt0, rt1, g0, g1, after)

            if not THR_GIVEN:

                @pl.when(anyr > 0)
                def _():
                    if ONLY4:
                        thr(4)
                    else:

                        @pl.when(tot <= tiles.thr_switch)
                        def _():
                            thr(4)

                        @pl.when(tot > tiles.thr_switch)
                        def _():
                            thr(2)

                @pl.when(anyr == 0)
                def _():
                    set_short(rt0, rt1)

            pw.mark("I_union")
            if not UNION_GIVEN:
                union_upto(bw0, bw1, jnp.int32(1 << 30))
            pw.end()
            return row0

        st[4] = jnp.int32(0)
        if THR_GIVEN or E.xout is not None:
            fp = jnp.clip(
                jnp.where(
                    l < cfg.idx_skip,
                    l,
                    cfg.idx_skip + (l - cfg.idx_skip) // cfg.idx_freq,
                ),
                0,
                max(E.Lf, 1) - 1,
            )
        if (
            THR_GIVEN and not UNION_GIVEN
        ):  # thr_given: this layer's recorded T / P / allowance -> b_row[2:5]
            cpy = pltpu.make_async_copy(
                E.xin["thr"].at[fp], row.at[pl.ds(2, 3)], E.misc_sem
            )
            cpy.start()
            cpy.wait()
        lax.fori_loop(0, nw, wave, jnp.int32(0))
        if (
            UNION_GIVEN
        ):  # union_given: the recorded union table (resolved entries, request
            for nm, dst in (
                ("uv", scr["own_uv"]),
                ("req", req),
                ("greq", scr["b_greq"]),
            ):  # and granule maps)
                cpy = pltpu.make_async_copy(E.xin[nm].at[fp], dst, E.misc_sem)
                cpy.start()
                cpy.wait()
            st[4] = scr["b_greq"][UR2 - 1]
        if (
            E.xout is not None
        ):  # rec build: export what the given arms load (output unchanged)
            scr["b_greq"][UR2 - 1] = st[
                4
            ]  # past every written granule (U <= UR_ = UR2 - XKC / 128)
            for nm, src in (
                ("thr", row.at[pl.ds(2, 3)]),
                ("uv", scr["own_uv"]),
                ("req", req),
                ("greq", scr["b_greq"]),
            ):
                cpy = pltpu.make_async_copy(src, E.xout[nm].at[fp], E.misc_sem)
                cpy.start()
                cpy.wait()
        U = st[4]
        st[0] = U
        pi.mark("I_udma")
        udma(U)
        pi.end()

    # ------------------------------------------------------------------ gather ring
    def issue(l, r):
        """Start granule r's row gathers into ring slot r % GR from the resolved source table (own_u):
        old entries from the current layer's caches, this step's own tokens from the VMEM staging rows."""
        b = scr["b_greq"][r]
        j = r - req[b, 1]
        nu = req[b, 0]
        n = jnp.clip(nu - 128 * j, 0, 128)
        n_old = jnp.clip(req[b, 2] - 128 * j, 0, n)
        slot = r % GR
        base = slot * 128
        own = scr["own_u"]
        gs = scr["g_sem"]

        def go(rn, rr):
            def old_one(k):
                v = own[r, k] & SRC_MASK
                page, sl = v >> 8, v & 255
                pltpu.make_async_copy(
                    rn.at[page, pl.ds(sl, 1)],
                    scr["g_nope"].at[pl.ds(base + k, 1)],
                    gs.at[slot, 0],
                ).start()
                pltpu.make_async_copy(
                    rr.at[page, pl.ds(sl >> 2, 1)],
                    scr["g_rope"].at[pl.ds(base + k, 1)],
                    gs.at[slot, 1],
                ).start()

            def u_body(i, c):
                for u in range(UNR):
                    old_one(i * UNR + u)
                return c

            nu16 = n_old // UNR
            lax.fori_loop(0, nu16, u_body, 0)

            def r_body(k, c):
                old_one(k)
                return c

            lax.fori_loop(nu16 * UNR, n_old, r_body, 0)

            def new_one(k, c):
                t2 = (own[r, k] & SRC_MASK) >> 2
                pltpu.make_async_copy(
                    E.wrow.at[pl.ds(t2, 1)],
                    scr["g_nope"].at[pl.ds(base + k, 1)],
                    gs.at[slot, 0],
                ).start()
                pltpu.make_async_copy(
                    E.rwrow.at[pl.ds(t2, 1)],
                    scr["g_rope"].at[pl.ds(base + k, 1)],
                    gs.at[slot, 1],
                ).start()
                return c

            lax.fori_loop(n_old, n, new_one, 0)

        if "gather" not in skip:
            csel(("nope", "rope"), l, go)

    def issue_part(l, r, part):
        """L2 `gearly` (stage B): granule r's OLD entries only (part "old": cache rows, issuable as soon as the
        layer's caches are known, i.e. a layer early) or its NEW entries only (part "new": this step's own tokens
        from the staging rows, which exist only after the layer's L_kvw). old + new = issue(l, r), same slots,
        same semaphores, same byte counts."""
        b = scr["b_greq"][r]
        j = r - req[b, 1]
        nu = req[b, 0]
        n = jnp.clip(nu - 128 * j, 0, 128)
        n_old = jnp.clip(req[b, 2] - 128 * j, 0, n)
        slot = r % GR
        base = slot * 128
        own = scr["own_u"]
        gs = scr["g_sem"]
        if part == "old":
            if (
                fault == "f_early_stale"
            ):  # FAULT: the early issue reads the PREVIOUS layer's caches
                l = jnp.maximum(l - 1, 0)

            def go(rn, rr):
                def old_one(k):
                    v = own[r, k] & SRC_MASK
                    page, sl = v >> 8, v & 255
                    pltpu.make_async_copy(
                        rn.at[page, pl.ds(sl, 1)],
                        scr["g_nope"].at[pl.ds(base + k, 1)],
                        gs.at[slot, 0],
                    ).start()
                    pltpu.make_async_copy(
                        rr.at[page, pl.ds(sl >> 2, 1)],
                        scr["g_rope"].at[pl.ds(base + k, 1)],
                        gs.at[slot, 1],
                    ).start()

                def u_body(i, c):
                    for u in range(UNR):
                        old_one(i * UNR + u)
                    return c

                nu16 = n_old // UNR
                lax.fori_loop(0, nu16, u_body, 0)

                def r_body(k, c):
                    old_one(k)
                    return c

                lax.fori_loop(nu16 * UNR, n_old, r_body, 0)

            if "gather" not in skip:
                csel(("nope", "rope"), l, go)
        else:

            def new_one(k, c):
                t2 = (own[r, k] & SRC_MASK) >> 2
                pltpu.make_async_copy(
                    E.wrow.at[pl.ds(t2, 1)],
                    scr["g_nope"].at[pl.ds(base + k, 1)],
                    gs.at[slot, 0],
                ).start()
                pltpu.make_async_copy(
                    E.rwrow.at[pl.ds(t2, 1)],
                    scr["g_rope"].at[pl.ds(base + k, 1)],
                    gs.at[slot, 1],
                ).start()
                return c

            if "gather" not in skip:
                lax.fori_loop(n_old, n, new_one, 0)

    GE = tuple(
        int(x) for x in getattr(tiles, "gearly", ()) or ()
    )  # (granules in the RS wait, ... + AG wait)
    if GE:
        assert len(GE) == 2 and 0 <= GE[0] <= GE[1] <= GR, (GE, GR)

    def early_issue(l_next, which):
        """Issue the old entries of the next (IndexShare) layer's granules [lo, hi) of the gather ring:
        which 0 -> [0, GE[0]), which 1 -> [GE[0], GE[1]) (clipped to U and GR)."""
        lo, hi = (0, GE[0]) if which == 0 else (GE[0], GE[1])
        U = st[0]

        def body(r, c):
            issue_part(l_next, r, "old")
            return c

        lax.fori_loop(jnp.minimum(lo, U), jnp.minimum(hi, U), body, 0)

    GREC_IN = getattr(E, "grec_in", None)
    GREC_OUT = getattr(E, "grec_out", None)
    GRECN = int(getattr(tiles, "grec", 0))

    def replay_issue(l, r):
        """Backing arm gather_replay: granule r's n gathered rows (nope and rope) DMA'd from the record of the
        clean program instead of 2n row DMAs; n rows exactly, so gwait's byte counts are unchanged."""
        b = scr["b_greq"][r]
        j = r - req[b, 1]
        n = jnp.clip(req[b, 0] - 128 * j, 0, 128)
        slot = r % GR
        base = slot * 128
        for s_, (src, dst) in enumerate(
            ((GREC_IN[0], scr["g_nope"]), (GREC_IN[1], scr["g_rope"]))
        ):

            @pl.when(n == 128)
            def _(s_=s_, src=src, dst=dst):
                pltpu.make_async_copy(
                    src.at[l, pl.ds(r * 128, 128)],
                    dst.at[pl.ds(base, 128)],
                    scr["g_sem"].at[slot, s_],
                ).start()

            for k in range(6, -1, -1):

                @pl.when(jnp.logical_and(n < 128, (n & (1 << k)) != 0))
                def _(s_=s_, src=src, dst=dst, k=k):
                    off = n & ~((2 << k) - 1)
                    pltpu.make_async_copy(
                        src.at[l, pl.ds(r * 128 + off, 1 << k)],
                        dst.at[pl.ds(base + off, 1 << k)],
                        scr["g_sem"].at[slot, s_],
                    ).start()

    def record_rows(l, r):
        """Record build: granule r's ring rows (after its wait) -> the record outputs (r < grec)."""
        slot = r % GR
        base = slot * 128

        @pl.when(r < GRECN)
        def _():
            for s_, (src, dst) in enumerate(
                ((scr["g_nope"], GREC_OUT[0]), (scr["g_rope"], GREC_OUT[1]))
            ):
                cp = pltpu.make_async_copy(
                    src.at[pl.ds(base, 128)],
                    dst.at[l, pl.ds(r * 128, 128)],
                    E.rec_sem.at[1],
                )
                cp.start()
                cp.wait()

    def gwait(r):
        b = scr["b_greq"][r]
        j = r - req[b, 1]
        n = jnp.clip(req[b, 0] - 128 * j, 0, 128)
        slot = r % GR
        if "gather" in skip:
            return
        for s_, buf in ((0, scr["g_nope"]), (1, scr["g_rope"])):

            @pl.when(n == 128)
            def _(s_=s_, buf=buf):
                pltpu.make_async_copy(
                    buf.at[pl.ds(0, 128)],
                    buf.at[pl.ds(0, 128)],
                    scr["g_sem"].at[slot, s_],
                ).wait()

            for k in range(6, -1, -1):

                @pl.when(jnp.logical_and(n < 128, (n & (1 << k)) != 0))
                def _(s_=s_, buf=buf, k=k):
                    pltpu.make_async_copy(
                        buf.at[pl.ds(0, 1 << k)],
                        buf.at[pl.ds(0, 1 << k)],
                        scr["g_sem"].at[slot, s_],
                    ).wait()

    # ------------------------------------------------------------------ attention
    lane_h = lax.broadcasted_iota(I32, (NH, 128), 1)
    row_q = lax.broadcasted_iota(I32, (QN, 1), 0) // NH
    lane128 = lax.broadcasted_iota(I32, (128, 128), 1)

    def qload(t0):
        """q of the request's QMAX rows -> qb [QN, KL] bf16 (row qq * NH + head) and qrb (rope dims in lanes
        0..63): per row, the 16 ranks' packed rows are read as one (R, QH) block and each head's 640 values are
        placed with sublane-strided stores (head = rank * HR + hh) into lane-blocked staging (b_sbuf, idle here)."""
        assert QA % 128 == 0 and HR % 2 == 0
        stg = scr["b_sbuf"]
        for qq in range(cfg.QMAX):
            t_ = jnp.minimum(t0 + qq, TMAX - 1)
            blk = scr["qall"][:, pl.ds(t_, 1), :].reshape(R, QH)  # [R, QH] u32
            full = jnp.concatenate(
                [unpack_lo(blk), unpack_hi(blk)], axis=1
            )  # [R, HR * QA] f32
            for hh in range(HR):
                for c in range(QA // 128):
                    dst = stg.at[0] if c < 4 else stg.at[1]
                    r_ = (c * NH if c < 4 else 0) + hh
                    dst[pl.ds(r_, R, stride=HR), :] = full[
                        :, hh * QA + c * 128 : hh * QA + (c + 1) * 128
                    ]
            r0 = qq * NH
            qn = jnp.concatenate(
                [stg[0, c * NH : (c + 1) * NH, :] for c in range(4)], axis=1
            )  # [NH, KL]
            scr["qb"][r0 : r0 + NH, :] = qn.astype(BF16)
            scr["qrb"][r0 : r0 + NH, :] = jnp.where(
                lane_h < cfg.DR, stg[1, 0:NH, :], 0.0
            ).astype(BF16)

    def reset():
        scr["m_s"][...] = jnp.full((QN, 1), NEG, F32)
        scr["l_s"][...] = jnp.zeros((QN, 1), F32)
        scr["att_acc"][...] = jnp.zeros((QN, KL), F32)

    def dec8(b):
        """fp8 byte (in i32) -> bf16, without the NaN-code fix. Valid gathered rows are written cache rows (writes
        clip to +-448: never a NaN code). A partial granule's tail rows (>= the request's entry count) are NOT
        gathered this layer and hold stale bytes: on the Mosaic path g_nope / g_rope are epool views overlapping
        expert slots 2-3, so after a MoE layer those rows hold expert-weight bytes (any byte, incl. the e4m3fn NaN
        codes 0x7F / 0xFF), and at step start whatever VMEM held. `front` therefore zeroes the nope words of the
        tail rows before decoding them (p = 0 x stale NaN would otherwise put NaN into P.V); stale rope words only
        reach scores of masked columns, which the NEG select replaces."""
        return lax.bitcast_convert_type(b.astype(U8), FP8).astype(BF16)

    def front(r, buf, ks):
        """Granule r, independent of the softmax state: decode the nope rows into kvb[buf], scores (4 nope +
        1 realigned-rope matmuls), scale, mask -> sbuf[buf] (masked entries = NEG)."""
        b = scr["b_greq"][r]
        j = r - req[b, 1]
        nu = req[b, 0]
        slot = r % GR
        if fault == "f_wrap":  # FAULT: after the ring wraps, read the next slot
            slot = jnp.where(r >= GR, (r + 1) % GR, slot)
        base = pl.multiple_of(slot * 128, 128)
        w = lax.bitcast_convert_type(scr["g_nope"][pl.ds(base, 128), :], I32)
        if (
            fault != "f_alias_poison_nan_noguard"
        ):  # FAULT (with the kernel's 0xFF poison): no tail zeroing
            w = jnp.where(
                j * 128 + sub128 < nu, w, 0
            )  # stale tail rows -> fp8 0 (dec8); valid rows unchanged
        for b_ in range(4):
            scr["kvb"][buf, :, b_ * 128 : (b_ + 1) * 128] = dec8(shr(w, 8 * b_) & 0xFF)
        s = jnp.zeros((QN, 128), F32)
        for b_ in range(4):
            s = s + lax.dot_general(
                scr["qb"][:, b_ * 128 : (b_ + 1) * 128],
                scr["kvb"][buf, :, b_ * 128 : (b_ + 1) * 128],
                NT,
                preferred_element_type=F32,
            )
        iv = scr["own_uv"][pl.ds(r, 1), :]  # [1, 128] entries
        seg = iv & 3  # rope quarter (local index & 3 == slot & 3)
        if fault == "ropeseg":
            seg = (seg + 1) & 3
        segc = jnp.transpose(jnp.broadcast_to(seg, (8, 128)))[:, 0:1]  # [128, 1]
        rw = lax.bitcast_convert_type(scr["g_rope"][pl.ds(base, 128), :], I32)
        A = jnp.where(
            segc == 0,
            rw,
            jnp.where(
                segc == 1,
                pltpu.roll(rw, 96, 1),
                jnp.where(segc == 2, pltpu.roll(rw, 64, 1), pltpu.roll(rw, 32, 1)),
            ),
        )
        W0 = A & 0xFF
        W1 = shr(pltpu.roll(A, 32, 1), 8) & 0xFF
        Wb = dec8(jnp.where(lane128 < 32, W0, jnp.where(lane128 < 64, W1, 0)))
        s = s + lax.dot_general(scr["qrb"][...], Wb, NT, preferred_element_type=F32)
        s = s * sm_scale
        if fault != "kscale":  # served order: s * sm_scale * k_scale (x 1.0 exact)
            s = s * ks
        if fault == "scale_twice":
            s = s * ks
        bits = shr(iv, MB_SH) & 15
        kk = j * 128 + lax.broadcasted_iota(I32, (1, 128), 1)
        mask = (((bits >> row_q) & 1) == 1) & (kk < nu)
        scr["b_sbuf"][buf] = jnp.where(mask, s, NEG)

    def back(buf):
        """Online-softmax update + PV for the granule in (sbuf, kvb)[buf]."""
        s = scr["b_sbuf"][buf]
        mask = s > NEG
        m_old = scr["m_s"][...]
        m_new = jnp.maximum(m_old, jnp.max(s, axis=1, keepdims=True))
        alpha = jnp.exp(m_old - m_new)
        p = jnp.where(mask, jnp.exp(s - m_new), 0.0)
        scr["l_s"][...] = alpha * scr["l_s"][...] + jnp.sum(p, axis=1, keepdims=True)
        if tiles.pv == "f32":
            pv = lax.dot_general(
                p, scr["kvb"][buf].astype(F32), NN, preferred_element_type=F32
            )
        else:
            pv = lax.dot_general(
                p.astype(BF16), scr["kvb"][buf], NN, preferred_element_type=F32
            )
        scr["att_acc"][...] = alpha * scr["att_acc"][...] + pv
        scr["m_s"][...] = m_new

    def back2():
        """Online-softmax update + PV for the two granules in buffers 0 and 1 at once: one rescale of the
        accumulator and one K = 256 PV pass per pair (the MXU sums the pair)."""
        s0 = scr["b_sbuf"][0]
        s1 = scr["b_sbuf"][1]
        m_old = scr["m_s"][...]
        m_new = jnp.maximum(
            m_old,
            jnp.maximum(
                jnp.max(s0, axis=1, keepdims=True), jnp.max(s1, axis=1, keepdims=True)
            ),
        )
        alpha = jnp.exp(m_old - m_new)
        p0 = jnp.where(s0 > NEG, jnp.exp(s0 - m_new), 0.0)
        p1 = jnp.where(s1 > NEG, jnp.exp(s1 - m_new), 0.0)
        scr["l_s"][...] = (
            alpha * scr["l_s"][...]
            + jnp.sum(p0, axis=1, keepdims=True)
            + jnp.sum(p1, axis=1, keepdims=True)
        )
        p = jnp.concatenate([p0, p1], axis=1)
        kv = jnp.concatenate([scr["kvb"][0], scr["kvb"][1]], axis=0)
        if tiles.pv == "f32":
            pv = lax.dot_general(p, kv.astype(F32), NN, preferred_element_type=F32)
        else:
            pv = lax.dot_general(p.astype(BF16), kv, NN, preferred_element_type=F32)
        scr["att_acc"][...] = alpha * scr["att_acc"][...] + pv
        scr["m_s"][...] = m_new

    def emit_row(t_, qq, empty, ks):
        r0 = qq * NH
        if empty:
            out = jnp.zeros((NH, KL), F32)
            lse = jnp.full((NH, 1), -jnp.inf, F32)
        else:
            lsum = scr["l_s"][r0 : r0 + NH, 0:1]
            mx = scr["m_s"][r0 : r0 + NH, 0:1]
            pv = scr["att_acc"][r0 : r0 + NH, :]
            if fault != "kscale":
                pv = pv * ks
            has = lsum > 0
            rl = jnp.maximum(lsum, 1e-30)
            inv = (
                pltpu.reciprocal(rl, approx=True)
                if getattr(tiles, "recip", "approx") == "approx"
                else 1.0 / rl
            )
            out = jnp.where(has, E.bf(pv * inv), 0.0)
            lse = jnp.where(has, mx + jnp.log(jnp.maximum(lsum, 1e-30)), -jnp.inf)
        lse_b = jnp.broadcast_to(lse, (NH, 128))
        if NH < 128:
            lse_b = jnp.concatenate(
                [lse_b, jnp.full((128 - NH, 128), -jnp.inf, F32)], axis=0
            )
        lse_row = jnp.transpose(lse_b)[0:1]
        if getattr(E, "debug", False):
            scr["dbgl"][pl.ds(t_, 1), :] = lse_row
            if "dbgo" in scr:
                scr["dbgo"][t_] = out
        # per owner d: heads 4d..4d+3 packed (lo: heads 4d, 4d+1; hi: 4d+2, 4d+3) | lse lanes (head - 4d):
        # sublane-strided loads give each head index hh's 16 owners at once (lane-blocked staging in b_sbuf)
        stg = scr["b_sbuf"]
        for c in range(4):
            stg[0, c * NH : (c + 1) * NH, :] = out[:, c * 128 : (c + 1) * 128]
        H = []
        for hh in range(HR):
            H.append(
                jnp.concatenate(
                    [stg[0, pl.ds(c * NH + hh, R, stride=HR), :] for c in range(4)],
                    axis=1,
                )
            )  # [R, KL]
        P = pack2(
            jnp.concatenate(H[: HR // 2], axis=1), jnp.concatenate(H[HR // 2 :], axis=1)
        )  # [R, OH]
        stg[1, 0:NH, :] = jnp.broadcast_to(lse, (NH, 128))
        lane_r = lax.broadcasted_iota(I32, (R, 128), 1)
        if LSE_TILE:
            # L4 C1: owner d's lse tile = rows d * HPAD + hh of `lsend`, lane = row t (one [HPAD, 128] f32 tile per
            # peer instead of 128 lse lanes in every payload row); the payload row is P alone
            tl = (t_ + 1) % TMAX if fault == "f_lse_tile_row" else t_
            for hh in range(HR):
                cur = scr["lsend"][pl.ds(hh, R, stride=HPAD), :]
                scr["lsend"][pl.ds(hh, R, stride=HPAD), :] = jnp.where(
                    lane_r == tl, stg[1, pl.ds(hh, R, stride=HR), :], cur
                )
            blk = P  # [R, OH]
        else:
            L16 = jnp.full((R, 128), -jnp.inf, F32)
            for hh in range(HR):  # lane hh of owner d's row = lse of head d * HR + hh
                L16 = jnp.where(lane_r == hh, stg[1, pl.ds(hh, R, stride=HR), :], L16)
            blk = jnp.concatenate(
                [P, lax.bitcast_convert_type(L16, U32)], axis=1
            )  # [R, OW]
        for d in range(R):
            scr["osend"][d, pl.ds(t_, 1), :] = blk[d : d + 1]

    def emit(b, empty, ks):
        t0, nq = req[b, 7], req[b, 8]
        for qq in range(cfg.QMAX):

            @pl.when(qq < nq)
            def _(qq=qq):
                emit_row(t0 + qq, qq, empty, ks)

    def attention(l, fresh, q_wait, after_req=None, early=None):
        """The all-request gather ring + attention: issue granules [0, min(U, GR)) (IndexShare layers:
        before the q all-gather wait), then per request: q, then per granule r of it: wait, compute,
        issue r + GR into the freed slot; then emit. after_req(b) (L4 C2: the chunked A2A) runs after request b's
        rows are emitted.
        early (L2 `gearly`, traced bool): the old entries of granules [0, GE[1]) were issued during the previous
        layer's AR#2 (`early_issue`); only their new entries are issued here."""
        del fresh  # one issue path: sources are resolved per fresh layer
        ks = E.ks()  # this layer's k_scale, [1, 1] f32
        U = st[0]
        pa = Phases(tiles.trace)
        pa.mark("A_issue")

        if GREC_IN is not None:

            def issue_at(r):
                @pl.when(r < GRECN)
                def _():
                    replay_issue(l, r)

                @pl.when(r >= GRECN)
                def _():
                    issue(l, r)

            def pre(r, c):
                issue_at(r)
                return c
        elif GE and early is not None:

            def pre(r, c):
                done = jnp.logical_and(early, r < GE[1])

                @pl.when(done)
                def _():
                    issue_part(l, r, "new")

                @pl.when(jnp.logical_not(done))
                def _():
                    issue(l, r)

                return c
        else:

            def pre(r, c):
                issue(l, r)
                return c

        QM = int(getattr(tiles, "qmid", 0) or 0)
        qmid_fn = getattr(q_wait, "mid", None) if QM else None
        if qmid_fn is not None:  # pair q AG: forward to the sibling after QM granules
            n_pre = jnp.minimum(U, GR)
            lax.fori_loop(0, jnp.minimum(n_pre, QM), pre, 0)
            qmid_fn()
            lax.fori_loop(jnp.minimum(n_pre, QM), n_pre, pre, 0)
        else:
            lax.fori_loop(0, jnp.minimum(U, GR), pre, 0)
        pa.mark("A_qwait")
        q_wait()
        pa.mark("A_comp")

        def per_req(b, c):
            nrow = (req[b, 0] + 127) // 128
            r0 = req[b, 1]

            @pl.when(req[b, 10] > 0)
            def _():
                @pl.when(nrow == 0)
                def _():
                    emit(b, True, ks)

                @pl.when(nrow > 0)
                def _():
                    qload(req[b, 7])
                    reset()
                    compute = "attn" not in skip

                    def wait_full(
                        r,
                    ):  # a granule before the request's last is always full
                        if "gather" not in skip:
                            for s_, buf_ in ((0, scr["g_nope"]), (1, scr["g_rope"])):
                                pltpu.make_async_copy(
                                    buf_.at[pl.ds(0, 128)],
                                    buf_.at[pl.ds(0, 128)],
                                    scr["g_sem"].at[r % GR, s_],
                                ).wait()

                    def refill(
                        r,
                    ):  # the slot of granule r is free once front(r) read it
                        @pl.when(r + GR < U)
                        def _():
                            if GREC_IN is not None:
                                issue_at(r + GR)
                            else:
                                issue(l, r + GR)

                    def fr(r, buf, last):
                        if last:
                            gwait(r)
                        else:
                            wait_full(r)
                        if GREC_OUT is not None:
                            record_rows(l, r)
                        if compute:
                            front(r, buf, ks)
                        refill(r)

                    npair = nrow // 2

                    def pbody(i, c2):  # granule pairs: two independent fronts, one back
                        r = r0 + 2 * i
                        fr(r, 0, False)
                        fr(r + 1, 1, False)
                        if compute:
                            back2()
                        return c2

                    # every pair but a last one that ends the request (its second granule may be partial)
                    lax.fori_loop(
                        0, jnp.where(nrow % 2 == 0, npair - 1, npair), pbody, 0
                    )

                    @pl.when(jnp.logical_and(nrow % 2 == 0, nrow > 0))
                    def _():
                        r = r0 + nrow - 2
                        fr(r, 0, False)
                        fr(r + 1, 1, True)
                        if compute:
                            back2()

                    @pl.when(nrow % 2 == 1)
                    def _():
                        fr(r0 + nrow - 1, 0, True)
                        if compute:
                            back(0)

                    emit(b, False, ks)

            if after_req is not None:
                after_req(b)
            return c

        lax.fori_loop(0, B_(), per_req, 0)
        pa.end()

    return dict(
        step_init=step_init,
        indexer_tail=indexer_tail,
        attention=attention,
        geometry=g,
        early_issue=early_issue,
    )
