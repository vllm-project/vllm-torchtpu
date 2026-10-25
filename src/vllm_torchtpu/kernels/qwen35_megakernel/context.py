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
"""Build-time geometry, VMEM/SMEM scratch layout, and runtime Ref binding context."""

from __future__ import annotations

import contextlib
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pool_alias
from .config import (
    ABLATIONS,
    AR0S_ON,
    ARPIPE_H1_ON,
    ARPIPE_OFF_B,
    ARPIPE_ON,
    ATT_GDEFER,
    ATT_KVDEFER,
    ATT_QSWAP,
    ATT_WGROUP,
    BF16,
    CONV_STAGE_MIN_B,
    CP_SH,
    DENSE,
    DMM_UNROLL,
    DMM_UNROLL_OFF_B,
    DWAIT_HOIST,
    EXPERT,
    F32,
    FLUSH_IN_AR1,
    FP8,
    GDN_LCV,
    GDN_SEQ_ROLL_MIN_B,
    GDN_ZDEFER,
    GDN_ZDEFER_ROLL,
    GDN_ZDEFER_ROLL_BUCKETS,
    GQ_PACK,
    I32,
    MKConfig,
    MXU_DR_BUCKETS,
    NN,
    O_NAMES,
    PRIM_NOW,
    ROUTE_SHRINK,
    S_NAMES,
    SH2_PRE_ROUTE,
    TOPK_LANES,
    Tiles,
    U8,
    U32,
    W_NAMES,
    WIDE_CAP,
    default_tiles,
    kernel_out_shapes,
    plan,
    rnd,
    rup,
    smalls_layout,
    weight_shapes,
)


# Wide step (A2) MoE phase switches. WIDE_DYN_ROW: expert e's routing-weight row by a dynamic
# sublane load of wnT_all (else a select-reduce over its rows). WIDE_SCATTER_TN: the one-hot
# scatter-add as a transposed-LHS dot_general (else an explicit transpose then a plain matmul).
WIDE_DYN_ROW = True
WIDE_SCATTER_TN = True
# All-reduce buffer slots of a wide step. Consecutive all-reduces alternate slots: a chip that
# receives the last round-1 payload of all-reduce n later than another chip (the sender issues
# its three copies in order) is still reading ag_recv when the earlier chip's round-1 payload of
# all-reduce n + 1 lands, so that payload must go to the other slot. One slot is only safe when
# compute separates consecutive all-reduces, which the MoE all-reduces of the microbatches lack.
WIDE_AR_SLOTS = 2
# GatedDeltaNet input-state prefetch slots of a wide step's microbatch, and whether the routed
# expert tile reuses the idle microbatch buffers xb / xb2 / acc (else its own scratch).
WIDE_NGI = 1
WIDE_TILE_REUSE = True
# Bisect switch: False runs the expert loop but drops its output (part_all keeps the shared
# expert only), so a parity against the no-experts kernel shows whether the loop itself disturbs
# later phases.
WIDE_SCATTER = True


class MKBuildContext:
    """Holds derived geometry constants, VMEM/SMEM scratch layout, and bound runtime Refs."""

    def __init__(
        self,
        c: MKConfig,
        tiles: Tiles | None = None,
        *,
        interpret: Any = False,
        forced_routing: bool = False,
        skip: frozenset[str] | set[str] = frozenset(),
        scopes: bool | int = False,
        row_mask: bool = False,
        pool_gather: bool | None = None,
        gq_pack: bool | None = None,
        topk_lanes: bool | None = None,
        dmm_unroll: int | None = None,
        conv_stage_min_b: int | None = None,
    ):
        c.check()
        assert set(skip) <= set(ABLATIONS), skip
        tiles = default_tiles(c) if tiles is None else tiles
        # Wide step: the geometry below is the microbatch's; c_wide keeps the whole step's row
        # count for the operand shapes and the all-rows MoE buffers.
        c_wide = c
        c = c.microbatch()
        c.check()
        N_MB = c_wide.n_mb
        BW = N_MB * c.B  # rows of the wide step, the tail microbatch padded
        WIDE_MB = c_wide.wide

        self.c = c
        self.c_wide = c_wide
        self.tiles = tiles
        self.interpret = interpret
        self.forced_routing = forced_routing
        self.skip = skip
        self.scopes = scopes
        self.row_mask = row_mask
        self.pool_gather = pool_gather
        self.gq_pack = GQ_PACK if gq_pack is None else gq_pack
        self.topk_lanes = TOPK_LANES if topk_lanes is None else topk_lanes
        self.dmm_unroll = DMM_UNROLL if dmm_unroll is None else dmm_unroll
        self.DMM_UNROLL_OFF_B = DMM_UNROLL_OFF_B
        self.GDN_LCV = GDN_LCV
        self.GDN_ZDEFER = GDN_ZDEFER
        self.ATT_GDEFER = ATT_GDEFER
        self.ATT_KVDEFER = ATT_KVDEFER
        self.ATT_QSWAP = ATT_QSWAP
        self.FLUSH_IN_AR1 = FLUSH_IN_AR1
        self.SH2_PRE_ROUTE = SH2_PRE_ROUTE
        self.ARPIPE_ON = ARPIPE_ON
        self.ARPIPE_OFF_B = ARPIPE_OFF_B
        self.ARPIPE_H1_ON = ARPIPE_H1_ON
        self.AR0S_ON = AR0S_ON
        self.MXU_DR_BUCKETS = MXU_DR_BUCKETS
        self.ROUTE_SHRINK = ROUTE_SHRINK
        self.CP_SH = CP_SH
        self.PRIM_NOW = PRIM_NOW
        self.ATT_WGROUP = ATT_WGROUP
        self.DWAIT_HOIST = DWAIT_HOIST
        self.GDN_ZDEFER_ROLL = GDN_ZDEFER_ROLL
        self.GDN_ZDEFER_ROLL_BUCKETS = GDN_ZDEFER_ROLL_BUCKETS
        self.GDN_SEQ_ROLL_MIN_B = GDN_SEQ_ROLL_MIN_B
        self.rnd = rnd
        self.conv_stage_min_b = (
            CONV_STAGE_MIN_B if conv_stage_min_b is None else conv_stage_min_b
        )

        R, B, M, H, L, E = c.R, c.B, c.M, c.H, c.L, c.E
        AE, MR, G = c.att_every, c.MR, c.G
        MB = MR if tiles.mb8 == 2 else (max(16, MR) if tiles.mb8 else M)
        p = plan(c, tiles)
        tk, nt, kinds = p["tk"], p["nt"], p["kinds"]
        t_gdn, t_att, t_block = p["t_gdn"], p["t_att"], p["t_block"]
        # dense tiles of one microbatch's mixer (input and output projections); a wide step
        # streams them once per microbatch and the MoE dense tiles once per layer
        MB_GDN = nt["lin_in"] + nt["lin_out"]
        MB_ATT = nt["att_in"] + nt["att_out"]
        if WIDE_MB:
            t_gdn = N_MB * MB_GDN + (t_gdn - MB_GDN)
            t_att = N_MB * MB_ATT + (t_att - MB_ATT)
            t_block = (AE - 1) * t_gdn + t_att
        T_TOTAL = (L // AE) * t_block
        lay, S = smalls_layout(c)
        NBD, NBE = tiles.nbd, tiles.nbe
        BROWS, EBROWS = tiles.bank // 128, tiles.ebank // 128
        T13, T2 = nt["e13"], nt["e2"]
        TE = T13 + T2
        NSS = 3 + (NBE - 1) // TE  # expert scale-row slots: in flight, plus the one being used
        PIPE = tiles.pipe and T13 == 1 and T2 == 1
        # EP multi-tile software pipeline: T13 = 2 * T2 (8 vs 4) and NBE >= TE (12 >= 12)
        PIPE_EP = (
            tiles.pipe
            and not PIPE
            and T13 == 2 * T2
            and NBE >= TE
            and NBE % TE == 0
        )
        BLK, NKV = tiles.kv_blk, tiles.nkv
        PAIR_KV = tiles.pair_kv
        if PAIR_KV:  # a pair waits on 2 stages: keep the next pair's 2 stages in flight meanwhile
            NKV = max(NKV, 4)
        assert (c.paged_kv or c.ctx % BLK == 0) and BLK % 128 == 0
        if c.paged_kv:
            assert BLK % c.page_size == 0, (BLK, c.page_size)
        nh, kh, vh, HD, KD, VD = c.nh, c.kh, c.vh, c.HD, c.KD, c.VD
        QKV, Z, BA, IR, ER = c.QKV, c.Z, c.BA, c.IR, c.ER
        PR, PC = c.PR, c.PC
        HPC = H // PC
        CH = min(1024, H)  # output lanes per down-projection chunk (single-tile experts)
        assert H % CH == 0
        NACC = max(c.NLI, c.NAI, H, E, 2 * c.ISR, 2 * IR)
        NY = max(c.NLI, c.NAI)
        KX2 = max(Z, nh * HD, c.ISR, IR)
        NMIX = max(Z, nh * HD)
        typed = interpret
        # Lazy dense refills: the attention block's and the MoE block's dense tiles are refilled
        # after the KV / expert stream instead of as each is consumed, so that the refills overlap
        # the all-reduces rather than compete for HBM. Needs the block's tiles to fit the ring at
        # once.
        N_MOE = nt["router"] + nt["sh13"] + nt["sh2"]
        LAZY_ATT = tiles.lazy and nt["att_in"] + nt["att_out"] <= NBD
        LAZY_MOE = tiles.lazy and N_MOE <= NBD
        # sequence-split attention: row b's KV block t is streamed by group rank (t + b) % G (TS=G)
        SPLIT = tiles.split and G > 1
        assert not SPLIT or G & (G - 1) == 0  # groups are aligned rank blocks: peers are me ^ off
        # DCP: the group's dcp shards each hold 1 / dcp of the pages, so the split runs over the
        # G // dcp ranks of one shard (every local item is this rank's at dcp == G); the merge
        # over the G partials is the same
        DCP = c.dcp
        assert DCP == 1 or SPLIT, "dcp > 1 needs the sequence-split attention (its merge)"
        TS = (G // DCP) if SPLIT else 1
        QR = rup(G * nh, 8) if SPLIT else 8  # query rows per sequence row in the KV loop
        # verify (T > 1): the KV rows are the NS sequences; each KV block is scored for the T * QR
        # query rows of its sequence (token t's heads in rows t * QR ..). T = 1: NS = B, TQ = QR.
        T, NS = c.T, c.NS
        VER = T > 1
        TQ = T * QR
        assert not (VER and tiles.inplace), "verify returns per-token states: no aliasing"
        SUB = tiles.kv_sub or BLK  # KV tokens per online-softmax update
        assert BLK % SUB == 0 and SUB % 128 == 0, (BLK, SUB)
        PW = HD + 128  # partial-softmax slab lanes: acc | m (64 lanes) | l (64 lanes)
        eff_ctx = max(c.ctx, c.max_pages_per_seq * c.page_size) if c.paged_kv else c.ctx
        max_nb = -(-eff_ctx // BLK)
        NWL = rup(NS * -(-max_nb // TS), 8)  # KV work items per rank, at most
        # above 16 rows ("wide"; up to 16 rows the kernel is unchanged):
        # - GCH: each GatedDeltaNet layer's states stream through NRB buffers of CB rows
        # - WAR: all-reduce on unfolded [MR, H] rows when WIDE or MR * H <= PR * PC (no row fold)
        FUSED = c.paged_kv and c.fused_pool_io
        WIDE = B > 16
        GCH = WIDE and tiles.gdn_rows > 0 and c.NGDN > 0 and not FUSED
        CB = tiles.gdn_rows if GCH else B
        assert not GCH or CB % 8 == 0, ("gdn_rows must be a multiple of 8", tiles.gdn_rows)
        assert not (GCH and VER) or CB % T == 0, ("gdn_rows must be a multiple of T", CB, T)
        NF, TR = divmod(B, CB)
        NCH = NF + (TR > 0)
        NRB = min(tiles.gdn_bufs, NCH)
        assert not GCH or NCH == 1 or NRB >= 2, ("gdn_bufs >= 2", tiles.gdn_bufs)
        CBV = CB * vh  # state heads per full chunk
        # GPR (gather-only pool fusion, 3-stage path): GatedDeltaNet input states are read straight
        # from the checkpoint pages of raw[] (read-only, through ckpt_indices / read_offsets) into
        # VMEM staging pages and unpacked there, instead of from a gathered HBM conv / rec copy.
        GPR_OK = c.paged_kv and not FUSED and GCH and VER
        GPR = GPR_OK if pool_gather is None else (bool(pool_gather) and GPR_OK)
        assert not pool_gather or GPR, ("pool_gather needs the paged chunked verify path", B, T)
        assert not GPR or c.page_size == 128, c.page_size
        GCONV_ROWS = (c.HR * QKV * 2) // (4 * 128)  # fp8 (4,128)-rows holding HR bf16 conv rows
        assert not GPR or (c.HR * QKV * 2) % (4 * 128) == 0, (c.HR, QKV)
        assert not GPR or GCONV_ROWS <= 64, GCONV_ROWS
        NGI = min(NS, 1 if B > 96 else (2 if B > 80 else (4 if B > 64 else 8)))
        NGO = min(NS, 1 if B > 64 else 2)
        if WIDE_MB:  # the microbatch buffers share VMEM with the all-rows MoE buffers
            NGI, NGO = min(NS, WIDE_NGI), 1
            # the mixer blocks take a whole input projection from the ring at once (GDEF, ZDEF
            # and the dtake_group paths), so a tile's bank must not be reused before that
            assert NBD >= max(nt["att_in"], nt["lin_in"]), (NBD, nt["att_in"], nt["lin_in"])
        G_PER_ATT = max(1, AE - 1)
        qkv_128 = QKV // 128
        A_conv = 16
        conv8_rows = A_conv * qkv_128
        ssm_u32_rows = (vh * KD) // 2
        REC_PAGES = c.rec_pages
        GDN_PAGES = c.gdn_pages
        GDN_STRIDE = c.gdn_stride
        SSM_U32_CAP = REC_PAGES * 128
        ckpt_is_3d = not c.ckpt_2d
        roff_is_2d = not c.roff_1d
        K_ckpt = c.ckpt_k if c.ckpt_k > 0 else T
        ckpt_t_off = 0 if K_ckpt == T else 1
        HIER = tiles.hier if R > 1 else 0
        N_HIER3 = max(1, R.bit_length() - 1)
        RSAG = tiles.rsag and R > 1 and HIER == 0
        HR = bool(getattr(tiles, "hr", 0)) and RSAG and R == 8
        WAR = (
            (WIDE or MR * H <= PR * PC or (RSAG and (PC // R) % 128 != 0))
            if tiles.war < 0
            else bool(tiles.war)
        )
        AR_R = MR if WAR else PR
        AR_C = H if WAR else PC
        CW = AR_C // R  # rsag: payload lanes summed by each rank
        HC2 = AR_C // 2  # hier=2: payload lanes per on-chip core
        assert not RSAG or CW % 128 == 0
        assert HIER in (0, 1, 2, 3) and (HIER == 0 or R == 8 or (HIER == 3 and R >= 2))
        assert HIER != 2 or HC2 % 128 == 0
        HX = bool(getattr(tiles, "hx", 0)) and HIER == 2 and R == 8 and (HC2 // 4) % 128 == 0
        ARP_COMMON = bool(
            B not in self.ARPIPE_OFF_B
            and FUSED
            and VER
            and AE > 1
            and c.NGDN > 0
            and tk["att_in"] == tk["lin_in"]
            and nt["lin_in"] == nt["att_in"]
            and nt["lin_in"] <= min(self.dmm_unroll, NBD)
            and B not in self.DMM_UNROLL_OFF_B
            and HC2 % tk["lin_in"] == 0
            and AR_C % tk["lin_in"] == 0
            and HC2 % 128 == 0
        )
        ARP_H1 = bool(self.ARPIPE_H1_ON and HIER == 1 and ARP_COMMON)
        A0_CW = min(512, HC2)
        AR0S = bool(
            self.AR0S_ON
            and R > 1
            and (HIER == 2 or (HIER == 1 and ARP_H1))
            and HC2 % A0_CW == 0
            and A0_CW % 128 == 0
            and max(nt["lin_out"], nt["att_out"]) <= min(self.dmm_unroll, NBD)
            and B not in self.DMM_UNROLL_OFF_B
        )
        AR_ROUNDS = (
            tuple(range(N_HIER3))
            if HIER == 3
            else (
                (0, 1, 2, 3)
                if HX
                else (0, 1, 2)
                if (HIER == 2 or ARP_H1)
                else ((0, 1, 2, 3) if HR else (0, 1) if (RSAG or HIER == 1) else (0,))
            )
        )
        AR_PEERS = 1 if HIER == 3 else (3 if (HIER in (1, 2) or HR) else max(R - 1, 1))
        ARP_TK = tk["lin_in"]
        ARPIPE = bool(self.ARPIPE_ON and ARP_COMMON and (HIER == 2 or ARP_H1))
        ARP_HPT = HC2 // ARP_TK if ARPIPE else 0
        ARP_S = (
            [j * (AR_C // ARP_TK) + i for j in range(H // AR_C) for i in range(ARP_HPT)]
            if ARPIPE
            else []
        )
        ARP = bool(ARPIPE and R > 1 and "ar" not in skip and "dense" not in skip
                   and not WIDE_MB)
        NSL = WIDE_AR_SLOTS if WIDE_MB else 2  # all-reduce payload buffer slots
        assert NSL in (1, 2) and (NSL == 2 or not ARP)
        LCV_NH = 3 * qkv_128
        LCV_NH16 = -(-LCV_NH // 16) * 16
        LCV = bool(
            self.GDN_LCV
            and FUSED
            and c.NGDN > 0
            and QKV % 128 == 0
            and KD == 128
            and VD == 128
            and QKV == 2 * kh * KD + vh * VD
            and LCV_NH16 <= conv8_rows
        )
        LCV_ROWS = -(-max((T + 3) * qkv_128, LCV_NH16) // 8) * 8
        # - PACK: the split attention's per-row slabs pack PB rows per 8-row tile
        PACK = (WIDE if tiles.pack < 0 else bool(tiles.pack)) and SPLIT and nh < 8
        PB = 8 // nh if PACK else 1
        NBP = -(-B // PB)

        # QV: att_qk / att_new work on [B, HD] row blocks (one per head) instead of per row.
        NHH = HD // 128
        SR = 8 // PB  # flattened rows per batch row (PACK: PB * nh == 8, so b * SR + h == qrow)
        QV = (SPLIT or VER) and HD % 128 == 0 and nh <= 8 and (not PACK or PB * nh == 8)
        # GQ (split verify, GQA query-group packing)
        QG = T * nh
        GQ = bool(self.gq_pack and QV and SPLIT and VER and G * nh == QR and QG % 16 == 0
                  and NS <= 64 and HD == 256)
        MLW = 128 // NS if GQ else 0  # lanes per sequence field (m in lanes 0.., l in 128..)
        if GQ:
            SR = nh
        QVR = B * nh + 8 if GQ else NBP * 8  # flattened QV rows (+8: strided-access headroom)

        scratch: dict[str, Any] = {
            "dsem": pltpu.SemaphoreType.DMA((NBD,)),
            "esem": pltpu.SemaphoreType.DMA((NBE,)),
            "sm": pltpu.VMEM((2, 1, S), F32),
            "sm_sem": pltpu.SemaphoreType.DMA((2,)),
            "res": pltpu.VMEM((MB, H), F32),
            "xb": pltpu.VMEM((MB, H), BF16),  # normed rows: mixer / router / expert input
            "xb2": pltpu.VMEM((MB, KX2), BF16),  # intermediate rows: out / down projection input
            "acc": pltpu.VMEM((MB, NACC), F32),
            **(
                {"acc13": pltpu.VMEM((MB, 2 * IR), F32)}
                if PIPE_EP
                else {}
            ),
            "ybuf": pltpu.VMEM((MB, NY), F32),
            "part": pltpu.VMEM((MB, H), F32),
            "mix": pltpu.VMEM((MB, NMIX), F32),
            # all-reduce [slot, ...]; rsag: lane chunks [slot, rank, AR_R, CW]
            "send": pltpu.VMEM((N_HIER3, NSL, AR_R, AR_C) if HIER == 3 else
                               (NSL, 2, AR_R, HC2) if HIER == 2 else
                               (NSL, R, AR_R, CW) if RSAG else (NSL, AR_R, AR_C), BF16),
            "recv": pltpu.VMEM((N_HIER3, NSL, 2, AR_R, AR_C) if HIER == 3 else
                               (NSL, 2, AR_R, HC2) if HIER == 2 else
                               (NSL, 2, AR_R, AR_C) if HIER == 1 else
                               (NSL, R, AR_R, CW) if RSAG else (NSL, R, AR_R, AR_C), BF16),
            "ar_ssem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), NSL, AR_PEERS)),
            "ar_rsem": pltpu.SemaphoreType.DMA((len(AR_ROUNDS), NSL, AR_PEERS)),
            # GatedDeltaNet
            "recb": pltpu.VMEM((B * vh, KD, VD), BF16),  # key-major states
            "sbuf": pltpu.VMEM((vh, KD, VD), F32),
            # T = 1: conv taps [3 x B rows], updated in place. T > 1: per-sequence history
            "convb": pltpu.VMEM((NS, c.HR, QKV) if VER else (c.CR, QKV), F32),
            "st_sem": pltpu.SemaphoreType.DMA((4,)),  # conv in, rec in, conv out, rec out
            # attention: KV ring stage s holds one work item; item i is block wl_t[i] of KV row
            # (sequence) wl_b[i] (item count in st[1])
            "kbuf": pltpu.VMEM((NKV, BLK, HD), BF16),
            "vbuf": pltpu.VMEM((NKV, BLK, HD), BF16),
            "kv_sem": pltpu.SemaphoreType.DMA((2, NKV)),
            "wl_b": pltpu.SMEM((NWL,), I32),
            "wl_t": pltpu.SMEM((NWL,), I32),
            "qb": pltpu.VMEM((NS, TQ, HD), BF16),
            # online-softmax state per KV row: running max (lane-broadcast), sum, unnormalized out
            "am": pltpu.VMEM((NS, TQ, 128), F32),
            "al": pltpu.VMEM((NS, TQ, 128), F32),
            "aacc": pltpu.VMEM((NS, TQ, HD), F32),
            "knew": pltpu.VMEM((MR, HD), F32),
            "vnew": pltpu.VMEM((MR, HD), F32),
            "gate": pltpu.VMEM((MR, nh * HD), F32),
            # [cos | sin, row, lane < ROT // 2]: the B rows' RoPE tables (positions are invariant)
            "rope": pltpu.VMEM((2, MR, 128), F32),
            "rcs": pltpu.VMEM((2, MR, c.ROT // 2), F32),  # per-row rope cos / sin (non-QV)
            "kv_out_sem": pltpu.SemaphoreType.DMA((2,)),
            # MoE
            "wn": pltpu.VMEM((MR, E), F32),  # renormalized routing weights of rows 0..MR-1
            "sg": pltpu.VMEM((MB, 128), F32),
            "hitv": pltpu.VMEM((MR, 128), I32),  # row b lanes 0..TOPK-1: row b's picks
            "hits": pltpu.SMEM((MR, 128), I32),
            "elist": pltpu.SMEM((rup(B * c.TOPK),), I32),  # deduplicated hit experts, pick order
            "seen": pltpu.SMEM((8, 128), I32),  # [e // 128, e % 128]: layer + 1 of e's last list
            "misc_sem": pltpu.SemaphoreType.DMA(()),
            "es": pltpu.VMEM((NSS, 1, 2 * IR + H), F32),
            "es_sem": pltpu.SemaphoreType.DMA((NSS,)),
            "st": pltpu.SMEM((8,), I32),
            "fn": pltpu.VMEM((1, H), F32),
        }
        KV_ITEM = 2 * 64 * BLK * 4
        KVE = bool(c.paged_kv and tiles.kv_ering and not typed and NBE >= NKV
                   and KV_ITEM <= tiles.ebank and HD == 256)
        NKVR = NBE if KVE else NKV
        if c.paged_kv:
            del scratch["kbuf"], scratch["vbuf"]
            if not KVE:
                scratch["kvbuf"] = pltpu.VMEM((NKV, 2, 64, 4, BLK), FP8)
            scratch["kv_sem"] = pltpu.SemaphoreType.DMA((NKVR,))
            scratch["knew_o"] = pltpu.VMEM((MR, HD), FP8)
            scratch["vnew_o"] = pltpu.VMEM((MR, HD), FP8)
            # clamped page ids of each KV work item (P_BLK per item), built once per step
            scratch["wl_kp"] = pltpu.SMEM((NWL * (BLK // c.page_size),), I32)
        if FUSED:
            for k in ("recb", "convb", "st_sem", "kv_out_sem"):
                del scratch[k]
            if c.NGDN > 0:
                scratch.update({
                    "gdn_in_page": pltpu.VMEM((NGI, GDN_PAGES, 2, 64, 4, c.page_size), FP8),
                    "gdn_out_page": pltpu.VMEM((NGO, GDN_PAGES * T, 2, 64, 4, c.page_size), FP8),
                    "gdn_in_sem": pltpu.SemaphoreType.DMA((NGI,)),
                    "gdn_out_sem": pltpu.SemaphoreType.DMA((NGO, T)),
                    "gdn_in_pend": pltpu.SMEM((NGI,), I32),
                    "gdn_out_mode": pltpu.SMEM((NGO,), I32),
                })
                if LCV:
                    scratch["gdn_cq"] = pltpu.VMEM((LCV_ROWS, 128), F32)
            else:
                del scratch["sbuf"]
            scratch.update({
                "kv_scat_page": pltpu.VMEM((NS, 2, 64, 4, c.page_size), FP8),
                "kv_rd_sem": pltpu.SemaphoreType.DMA((NS,)),
                "kv_wr_sem": pltpu.SemaphoreType.DMA((NS,)),
                "kv_rd_pend": pltpu.SMEM((NS,), I32),
                "kv_wr_pend": pltpu.SMEM((NS,), I32),
            })
        ECOPY = not c.moe_tp and c.e_slots > ER
        assert not (row_mask and ECOPY), "row_mask: identity expert placement only"
        EXTRA = c.EW - ER
        if ECOPY:
            nb_picks = rup(B * c.TOPK)
            scratch.update({
                "ecopy": pltpu.SMEM((2, 1, E), F32),
                "ec_sem": pltpu.SemaphoreType.DMA((2,)),
                "wlist": pltpu.SMEM((nb_picks,), I32),
                "cplist": pltpu.SMEM((nb_picks,), I32),
                "cpchoice": pltpu.SMEM((nb_picks,), I32),
                "rload": pltpu.SMEM((R,), I32),
            })
            if tiles.ecopy_opt == 1:
                scratch["ulist"] = pltpu.SMEM((nb_picks,), I32)
        if SPLIT:
            scratch.update({
                "qg": pltpu.VMEM((G, B, 8, HD), F32),  # [g]: group rank g's query rows
                "pso": pltpu.VMEM((G, B, 8, PW), F32),  # [g]: partials of group rank g's heads
                "pri": pltpu.VMEM((G, B, 8, PW), F32),  # [g]: partials of own heads from rank g
                "sq_ssem": pltpu.SemaphoreType.DMA((G - 1,)),
                "sq_rsem": pltpu.SemaphoreType.DMA((G - 1,)),
                "sp_ssem": pltpu.SemaphoreType.DMA((G - 1,)),
                "sp_rsem": pltpu.SemaphoreType.DMA((G - 1,)),
            })
        elif VER:
            scratch["qg"] = pltpu.VMEM((1, B, 8, HD), F32)  # own query rows (bf16-exact)
        if VER and not GCH and not FUSED:
            scratch.update({
                "ub": pltpu.VMEM((NS, rup(T, 8), QKV), F32),  # [s, t]: token t's conv output
                "rin_sem": pltpu.SemaphoreType.DMA((NS,)),  # recurrent state in, per sequence
            })
        if HR:  # this rank's chunk sum; the 4 chips' partials of this rank's chunk
            scratch["ag_send"] = pltpu.VMEM((NSL, AR_R, CW), BF16)
            scratch["ag_recv"] = pltpu.VMEM((NSL, 4, AR_R, CW), BF16)
        elif RSAG:  # round 2: this rank's chunk sum, and the chunk sums gathered from every rank
            scratch["ag_send"] = pltpu.VMEM((NSL, AR_R, CW), BF16)
            scratch["ag_recv"] = pltpu.VMEM((NSL, R, AR_R, CW), BF16)
        elif HIER == 1:
            scratch["ag_send"] = pltpu.VMEM((NSL, AR_R, AR_C), BF16)
            scratch["ag_recv"] = pltpu.VMEM((NSL, 4, AR_R, AR_C), BF16)
        elif HIER == 2:
            scratch["ag_send"] = pltpu.VMEM((NSL, AR_R, HC2), BF16)
            scratch["ag_recv"] = pltpu.VMEM((NSL, 4, AR_R, HC2), BF16)
            scratch["s2_send"] = pltpu.VMEM((NSL, AR_R, HC2), BF16)
            scratch["s2_recv"] = pltpu.VMEM((NSL, 2, AR_R, HC2), BF16)
        if ARPIPE:
            scratch["arp"] = pltpu.VMEM((MB, 128), F32)
        if forced_routing:
            # wide: 128-row chunks, double-buffered. The next chunk's copy must not land in the
            # buffer a chunk's top-k is still reading (the loads are not complete when the copy
            # starts), so it goes to the other slot.
            scratch["rb"] = pltpu.VMEM((2, 128, E) if WIDE_MB else (2, M, E), F32)
            scratch["rb_sem"] = pltpu.SemaphoreType.DMA((2,))
        BWP = rup(BW, 128)  # all-rows buffers in 128-row chunks (the routing chunk)
        CAP = min(WIDE_CAP, MB)  # gathered rows per expert tile (the tile reuses xb's rows)
        if WIDE_MB:
            del scratch["part"], scratch["wn"]  # the MoE phase works on part_all / wnT_all
            scratch.update({
                # residual rows of every microbatch (bf16-exact values), the MoE input rows and
                # the MoE output accumulator of the whole step
                "res_all": pltpu.VMEM((BW, H), BF16),
                "xb_all": pltpu.VMEM((BWP, H), BF16),
                "part_all": pltpu.VMEM((BWP, H), F32),
                # routing weights transposed: expert e's row holds its weight for every row
                "wnT_all": pltpu.VMEM((E, BWP), F32),
                "io_sem": pltpu.SemaphoreType.DMA((2,)),
            })
            if not WIDE_TILE_REUSE:
                scratch.update({
                    "xgw": pltpu.VMEM((CAP, H), BF16),
                    "midw": pltpu.VMEM((CAP, IR), BF16),
                    "accw": pltpu.VMEM((CAP, max(2 * IR, H)), F32),
                })
            # the expert tile's gathered rows, intermediate and accumulators reuse the
            # microbatch buffers xb, xb2 and acc (idle during the MoE phase); the per-expert
            # row counts ride in hitv / hits rows MROW + 1 .. 2 * MROW
            assert CAP % 8 == 0 and KX2 >= IR and NACC >= H and MR >= 2 * (E // 128) + 1, (
                MB, CAP, KX2, IR, NACC, H, MR, E)
        if GCH:  # [buffer, CB * vh, KD, VD]: one row chunk of a layer's key-major states
            scratch["recb"] = pltpu.VMEM((NRB, CBV, KD, VD), BF16)
            scratch["rec_sem"] = pltpu.SemaphoreType.DMA((2, NRB))  # [in / out, buffer]
            if GPR:
                scratch["gpr_rec"] = pltpu.VMEM(
                    (NRB, CB // T, REC_PAGES, 2, 64, 4, c.page_size), FP8)
                scratch["gpr_conv"] = pltpu.VMEM((NS, GCONV_ROWS, 4, c.page_size), FP8)
                scratch["convb16"] = pltpu.VMEM((NS, c.HR, QKV), BF16)
        elif WIDE and c.NGDN == 0 and not FUSED:  # attention-only stack: no GatedDeltaNet buffers
            for k in ("recb", "sbuf", "convb"):
                del scratch[k]
        if PACK:  # [g, tile]: PB rows' slabs per tile (qrow)
            scratch["qg"] = pltpu.VMEM((G, NBP, 8, HD), F32)
            scratch["pso"] = pltpu.VMEM((G, NBP, 8, PW), F32)
            scratch["pri"] = pltpu.VMEM((G, NBP, 8, PW), F32)
        if QV:  # row-vector layout: [g, lane chunk j, row b * SR + head, 128 lanes]
            scratch["qg"] = pltpu.VMEM((G if SPLIT and not GQ else 1, NHH, QVR, 128), F32)
            scratch["ost"] = pltpu.VMEM((NHH, QVR, 128), F32)  # att_new: rnd(out) of row b, head h
        if GQ:
            del scratch["pso"]
            scratch.update({
                "qxb": pltpu.VMEM((G, NS, QG, HD), BF16),
                "psend": pltpu.VMEM((G, NS + 1, QG, HD), F32),
                "pri": pltpu.VMEM((G, NS + 1, QG, HD), F32),
            })
        vshape = {k: (tk[k], kinds[k][1]) for k in kinds}
        vdtype = {k: kinds[k][2] for k in kinds}
        for k in kinds:
            assert pool_alias.storage_size(vshape[k], jnp.dtype(vdtype[k])) <= (
                tiles.ebank if k in EXPERT else tiles.bank), k
        if typed:
            for k in DENSE:
                scratch["dring_" + k] = pltpu.VMEM((NBD,) + vshape[k], vdtype[k])
            for k in EXPERT:
                scratch["ering_" + k] = pltpu.VMEM((NBE,) + vshape[k], vdtype[k])
        else:
            scratch["dpool"] = pltpu.VMEM((NBD * BROWS, 128), U8)
            scratch["epool"] = pltpu.VMEM((NBE * EBROWS, 128), U8)
        s_names = list(scratch)
        if FUSED:
            st_names = ("raw", "page_table", "ckpt_indices", "read_offsets") + (
                ("active_seqs",) if c.act_mode == "ref" else ()
            )
            o_names = ("hidden", "raw") if bool(interpret) else ("hidden",)
        elif GPR:
            st_names = ("raw", "page_table", "ckpt_indices", "read_offsets")
            o_names = O_NAMES
        elif c.paged_kv:
            st_names = ("conv", "rec", "raw", "page_table")
            o_names = O_NAMES
        else:
            st_names = S_NAMES
            o_names = O_NAMES
        n_w, n_s = len(W_NAMES), len(st_names)

        # Store all derived constants on self
        self.R, self.B, self.M, self.H, self.L, self.E = R, B, M, H, L, E
        self.AE, self.MR, self.G = AE, MR, G
        self.MB = MB
        self.p = p
        self.tk, self.nt, self.kinds = tk, nt, kinds
        self.t_gdn, self.t_att, self.t_block = t_gdn, t_att, t_block
        self.T_TOTAL = T_TOTAL
        self.lay, self.S = lay, S
        self.NBD, self.NBE = NBD, NBE
        self.BROWS, self.EBROWS = BROWS, EBROWS
        self.T13, self.T2, self.TE, self.NSS = T13, T2, TE, NSS
        self.PIPE, self.PIPE_EP = PIPE, PIPE_EP
        self.BLK, self.NKV, self.PAIR_KV = BLK, NKV, PAIR_KV
        self.nh, self.kh, self.vh = nh, kh, vh
        self.HD, self.KD, self.VD = HD, KD, VD
        self.QKV, self.Z, self.BA, self.IR, self.ER = QKV, Z, BA, IR, ER
        self.PR, self.PC, self.HPC, self.CH = PR, PC, HPC, CH
        self.NACC, self.NY, self.KX2, self.NMIX = NACC, NY, KX2, NMIX
        self.typed = typed
        from . import config as _cfg_late
        self.SH_AT = int(_cfg_late.SH_AT)
        # Serving knob (hostov stack e4244f756 / 6aa2cabb8): VLLM_TPU_QWEN35_MK_SH_AT=<0|1|2> for
        # buckets of at most VLLM_TPU_QWEN35_MK_SH_AT_MAX_ROWS rows (default 8). Unset = config.SH_AT.
        import os as _os_sh
        _sh_env = _os_sh.environ.get("VLLM_TPU_QWEN35_MK_SH_AT", "").strip()
        if _sh_env and int(self.B) <= int(_os_sh.environ.get("VLLM_TPU_QWEN35_MK_SH_AT_MAX_ROWS", "8")):
            self.SH_AT = int(_sh_env)
        if _sh_env:
            print(f"[mk] SH_AT={self.SH_AT} for bucket rows={int(self.B)} (env {_sh_env}, max rows "
                  f"{_os_sh.environ.get('VLLM_TPU_QWEN35_MK_SH_AT_MAX_ROWS', '8')})", flush=True)
        self.N_MOE, self.LAZY_ATT, self.LAZY_MOE = N_MOE, LAZY_ATT, LAZY_MOE
        self.SPLIT, self.TS, self.QR, self.DCP = SPLIT, TS, QR, DCP
        self.T, self.NS, self.VER, self.TQ = T, NS, VER, TQ
        self.SUB, self.PW, self.eff_ctx, self.max_nb, self.NWL = SUB, PW, eff_ctx, max_nb, NWL
        self.FUSED, self.WIDE, self.GCH = FUSED, WIDE, GCH
        self.CB, self.NF, self.TR, self.NCH, self.NRB, self.CBV = CB, NF, TR, NCH, NRB, CBV
        self.GPR_OK, self.GPR, self.GCONV_ROWS = GPR_OK, GPR, GCONV_ROWS
        self.NGI, self.NGO, self.G_PER_ATT = NGI, NGO, G_PER_ATT
        self.qkv_128, self.A_conv, self.conv8_rows, self.ssm_u32_rows = (
            qkv_128,
            A_conv,
            conv8_rows,
            ssm_u32_rows,
        )
        self.REC_PAGES, self.GDN_PAGES, self.SSM_U32_CAP = REC_PAGES, GDN_PAGES, SSM_U32_CAP
        self.GDN_STRIDE = GDN_STRIDE
        self.ckpt_is_3d, self.roff_is_2d = ckpt_is_3d, roff_is_2d
        self.K_ckpt, self.ckpt_t_off = K_ckpt, ckpt_t_off
        self.HIER, self.N_HIER3, self.RSAG, self.WAR = HIER, N_HIER3, RSAG, WAR
        self.HR, self.HX = HR, HX
        self.AR_R, self.AR_C, self.CW, self.HC2 = AR_R, AR_C, CW, HC2
        self.AR_ROUNDS, self.AR_PEERS = AR_ROUNDS, AR_PEERS
        self.ARP_COMMON, self.ARP_H1 = ARP_COMMON, ARP_H1
        self.A0_CW, self.AR0S = A0_CW, AR0S
        self.ARP_TK, self.ARPIPE, self.ARP_HPT, self.ARP_S, self.ARP = (
            ARP_TK,
            ARPIPE,
            ARP_HPT,
            ARP_S,
            ARP,
        )
        self.LCV, self.LCV_NH, self.LCV_NH16, self.LCV_ROWS = LCV, LCV_NH, LCV_NH16, LCV_ROWS
        self.PACK, self.PB, self.NBP = PACK, PB, NBP
        self.NHH, self.SR, self.QV = NHH, SR, QV
        self.QG, self.GQ, self.MLW, self.QVR = QG, GQ, MLW, QVR
        self.KV_ITEM, self.KVE, self.NKVR = KV_ITEM, KVE, NKVR
        self.ECOPY, self.EXTRA = ECOPY, EXTRA
        self.PRIM_NOW_K = bool(
            self.PRIM_NOW
            and self.ROUTE_SHRINK
            and tiles.route_vec
            and self.ECOPY
            and c.E % 128 == 0
            and c.E // 128 + 1 <= self.MR
            and c.R * self.EXTRA + c.R + 1 <= 128
            and c.E < (1 << self.CP_SH)
            and 128 % self.ER == 0
        )
        self.WIDE_MB, self.N_MB, self.BW, self.BWP, self.CAP = WIDE_MB, N_MB, BW, BWP, CAP
        self.MB_GDN, self.MB_ATT = MB_GDN, MB_ATT
        self.WIDE_DYN_ROW = WIDE_DYN_ROW
        self.WIDE_SCATTER_TN = WIDE_SCATTER_TN
        self.N_AR_SLOTS = NSL
        self.WIDE_TILE_REUSE = WIDE_TILE_REUSE
        self.WIDE_SCATTER = WIDE_SCATTER
        self.BLIVE = c_wide.B  # live rows of the wide step (the rest of the tail is padding)
        self.vshape, self.vdtype = vshape, vdtype
        self.scratch = scratch
        self.s_names = s_names
        self.st_names = st_names
        self.o_names = o_names
        self.n_w, self.n_s = n_w, n_s
        self.comm = R > 1 and "ar" not in skip

        # Precompute Pallas BlockSpecs, output shapes, aliases, and info dictionary
        oshapes = kernel_out_shapes(c_wide)
        if GPR:  # bf16 conv history output (exact: cached bf16 taps + bf16-rounded inputs)
            oshapes = {**oshapes, "conv": (oshapes["conv"][0], BF16)}
        w_specs = [
            tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.L))
            if (not c.stack_e and k in ("e13", "e2"))
            else pl.BlockSpec(memory_space=pltpu.HBM)
            for k in W_NAMES
        ]
        s_specs = [
            tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.NATT))
            if k == "raw"
            else pl.BlockSpec(memory_space=pltpu.SMEM)
            if k in ("page_table", "ckpt_indices", "read_offsets", "active_seqs")
            else pl.BlockSpec(memory_space=pltpu.HBM)
            for k in st_names
        ]
        hid_space = pltpu.HBM if WIDE_MB else pltpu.VMEM
        self.in_specs = (
            [pl.BlockSpec(memory_space=pltpu.SMEM), pl.BlockSpec(memory_space=hid_space)]
            + w_specs
            + s_specs
            + ([pl.BlockSpec(memory_space=pltpu.HBM)] if forced_routing else [])
            + ([pl.BlockSpec(memory_space=pltpu.SMEM)] if row_mask else [])
        )
        n_w_leaves = sum(c.L if (not c.stack_e and k in ("e13", "e2")) else 1 for k in W_NAMES)
        self.out_shape = [
            tuple(jax.ShapeDtypeStruct(*s) for s in oshapes[k])
            if k == "raw"
            else jax.ShapeDtypeStruct(*oshapes[k])
            for k in o_names
        ]
        self.out_specs = [
            pl.BlockSpec(memory_space=hid_space)
            if k == "hidden"
            else tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(c.NATT))
            if k == "raw"
            else pl.BlockSpec(memory_space=pltpu.HBM)
            for k in o_names
        ]
        if FUSED:
            self.aliases = (
                {2 + n_w_leaves + a: 1 + a for a in range(c.NATT)}
                if bool(interpret)
                else {}
            )
        else:
            self.aliases = (
                {2 + n_w_leaves + st_names.index(k): o_names.index(k) for k in ("conv", "rec")}
                if tiles.inplace
                else {}
            )
        vmem = 0
        for k in s_names:
            v = scratch[k]
            if hasattr(v, "shape") and "VMEM" in str(getattr(v, "memory_space", "")).upper():
                vmem += math.prod(v.shape) * jnp.dtype(v.dtype).itemsize
        self.vmem_scratch_bytes = vmem
        self.info = {
            "plan": p,
            "T_TOTAL": T_TOTAL,
            "vmem_scratch_bytes": vmem,
            "s_names": s_names,
            "wshape": weight_shapes(c, tiles),
            "NSS": NSS,
            "TE": TE,
        }

        # Runtime attributes populated by bind_refs(*refs) inside kernel(*refs)
        self.pos_ref: Any = None
        self.hid_ref: Any = None
        self.W: dict[str, Any] = {}
        self.ST: dict[str, Any] = {}
        self.rb_hbm: Any = None
        self.mr_ref: Any = None
        self.O: dict[str, Any] = {}
        self.out_ref: Any = None
        self.raw_refs: Any = None
        self.scr: dict[str, Any] = {}
        self.me: Any = None
        self.gi: Any = None
        self.dr: Any = None  # DCP rank in the KV-head group (gi % dcp)
        self.gs: Any = None  # split rank among the ranks holding the same shard (gi // dcp)
        self.arp_c: Any = None
        self.lane_e8: Any = None
        self.lane128: Any = None
        self.row8: Any = None
        self.row_m: Any = None
        self.live: Any = None
        self.EM: int = (1 << max(1, (E - 1).bit_length())) - 1
        self.FLT_MIN: float = float(jnp.finfo(F32).tiny)

    def arp_r(self, x, rows=None):
        """Scale x by scr['arp'][:, 0:1] when ARP is enabled."""
        if not self.ARP:
            return x
        return x * (self.scr["arp"][:, 0:1] if rows is None else self.scr["arp"][rows, 0:1])

    def sc(self, name: str, level: int = 1):
        """Return a named device trace scope context manager when scopes >= level."""
        return jax.named_scope(name) if int(self.scopes) >= level else contextlib.nullcontext()

    def mm(self, x, w, dims=NN):
        """Matrix multiply in f32 accumulation, casting w in interpret mode when dtypes differ."""
        if self.interpret and x.dtype != w.dtype:
            w = w.astype(x.dtype)
        return lax.dot_general(x, w, dims, preferred_element_type=F32)

    def qrow(self, b: int):
        """Row b's tile and first sublane in the per-row slabs (qg, pso, pri)."""
        return b // self.PB, (b % self.PB) * self.nh

    def sm(self, slot, name: str, n: int | None = None):
        """Slice smalls parameter `name` from SMEM/VMEM double buffer `slot`."""
        o, w = self.lay[name]
        return self.scr["sm"][slot, :, o : o + (w if n is None else n)]

    def rms_zc(self, x, w):
        """Zero-centered RMSNorm rounded to bf16 activation precision."""
        v = x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.c.eps)
        return rnd(v * (1.0 + w))

    def bind_refs(self, refs: tuple[Any, ...]) -> None:
        """Bind Pallas kernel operand/output/scratch Refs and initialize rank/iota tracers."""
        self.pos_ref, self.hid_ref = refs[:2]
        self.W = dict(zip(W_NAMES, refs[2 : 2 + self.n_w]))
        self.ST = dict(zip(self.st_names, refs[2 + self.n_w : 2 + self.n_w + self.n_s]))
        k_ = 2 + self.n_w + self.n_s
        self.rb_hbm = None
        if self.forced_routing:
            self.rb_hbm = refs[k_]
            k_ += 1
        self.mr_ref = None
        if self.row_mask:
            self.mr_ref = refs[k_]
            k_ += 1
        self.O = dict(zip(self.o_names, refs[k_ : k_ + len(self.o_names)]))
        self.out_ref = self.O["hidden"]
        self.raw_refs = (
            (self.O["raw"] if bool(self.interpret) else self.ST["raw"])
            if self.FUSED
            else (self.ST["raw"] if self.c.paged_kv else None)
        )
        self.scr = dict(zip(self.s_names, refs[k_ + len(self.o_names) :]))
        self.pos_all, self.hid_all, self.out_all = self.pos_ref, self.hid_ref, self.out_ref
        self.ST_all = dict(self.ST)
        self.me = lax.axis_index("tp").astype(I32)
        self.gi = lax.rem(self.me, self.G) if self.SPLIT else jnp.int32(0)
        self.dr = lax.rem(self.gi, self.DCP) if self.DCP > 1 else jnp.int32(0)
        self.gs = lax.div(self.gi, self.DCP) if self.DCP > 1 else self.gi
        self.arp_c = (
            (jnp.int32(0) if self.HIER == 1 else lax.rem(self.me, 2))
            if self.ARP
            else None
        )
        self.lane_e8 = lax.broadcasted_iota(I32, (self.MR, self.E), 1)
        self.lane128 = lax.broadcasted_iota(I32, (self.MR, 128), 1)
        self.row8 = lax.broadcasted_iota(I32, (self.MR, 1), 0)
        self.row_m = lax.broadcasted_iota(I32, (self.MB, 1), 0)
        self.live = self.row_m < self.B
        self.EM = (1 << max(1, (self.E - 1).bit_length())) - 1
        self.FLT_MIN = float(jnp.finfo(F32).tiny)

    def bind_microbatch(self, m) -> None:
        """Points the per-row and per-sequence operand refs at microbatch m of a wide step."""
        if not self.WIDE_MB:
            return
        nb, ns = self.c.B, self.c.NS
        r0 = pl.multiple_of(m * nb, nb)
        s0 = pl.multiple_of(m * ns, ns)
        self.pos_ref = self.pos_all.at[pl.ds(r0, nb)]
        st = dict(self.ST_all)
        if "page_table" in st:
            st["page_table"] = self.ST_all["page_table"].at[pl.ds(s0, ns)]
        if "ckpt_indices" in st:
            ck = self.ST_all["ckpt_indices"]
            st["ckpt_indices"] = ck.at[:, pl.ds(s0, ns)] if self.ckpt_is_3d else ck.at[pl.ds(s0, ns)]
        if "read_offsets" in st:
            ro = self.ST_all["read_offsets"]
            st["read_offsets"] = ro.at[:, pl.ds(s0, ns)] if self.roff_is_2d else ro.at[pl.ds(s0, ns)]
        if "active_seqs" in st:
            st["active_seqs"] = self.ST_all["active_seqs"].at[pl.ds(s0, ns)]
        self.ST = st
