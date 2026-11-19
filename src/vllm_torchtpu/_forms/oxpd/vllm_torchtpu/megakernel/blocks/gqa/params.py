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
"""`GQABlock` build-time parameters, tunables and their derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from jax.experimental.pallas import tpu as pltpu

from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec, NumericsClass
from ...fused_io_hook import fused_io
from ...ops import BF16, F32, FP8, I32, rup
from ...pool.config import cache_exec, resolved_max_pages_per_seq

NN = (((1,), (0,)), ((), ()))
NT = (((1,), (1,)), ((), ()))

# Geometry (layout) requirements, not per-bucket tuning:
# - KVE views each KV ring stage as `(2, HD // 4, BLK)` u32 rows of the untyped
#   expert pool; that view is validated for HD == 256 only (the Qwen3.5
#   layout). Other head dims keep the separate `kvbuf` ring.
# - GQ sends the partial-softmax `(m | l)` pair as two 128-lane halves of one
#   HD-wide row of the exchange buffer, so it needs HD == 2 * 128.
KVE_HD = 256
GQ_ML_LANES = 2 * 128


@dataclass(frozen=True)
class GQAParams:
    """Block-owned build-time parameters for `GQABlock` (`direct_hbm` & `paged_kv`)."""

    H: int
    nh: int
    HD: int
    ROT: int
    NAI: int
    G: int
    BLK: int
    NKV: int
    NKVR: int
    KVE: bool
    KV_ITEM: int
    EBROWS: int
    PAIR_KV: bool
    SUB: int
    AMASK: int
    SPLIT: bool
    TS: int
    QR: int
    TQ: int
    PW: int
    NWL: int
    PACK: bool
    PB: int
    NBP: int
    QV: bool
    GQ: bool
    NHH: int
    SR: int
    QG: int
    MLW: int
    QVR: int
    LAZY_ATT: bool
    eps: float
    theta: float
    zero_centered_norm: bool
    rope_fp32: bool = False

    def qrow(self, b: int) -> tuple[int, int]:
        return b // self.PB, (b % self.PB) * self.nh


@dataclass(frozen=True)
class GQATunables:
    """`GQABlock` knobs (`Tiles.blocks["gqa"]`); `NUMERICS` classes each field."""

    kv_blk: int = 2048  # KV tokens per block
    nkv: int = 3  # KV ring depth
    split: bool = True  # Sequence-split attention across G ranks sharing a KV head
    pack: int = -1  # Pack PB = 8 // nh rows per 8-row tile (-1: auto, WIDE buckets)
    kv_sub: int = 0  # KV tokens per online-softmax update inside a block (0: full)
    amask: int = 1  # Attention mask mode (0..4)
    kv_scatter_vec_ns: int = 16  # Non-paged host KV update: one scatter above this NS
    # Opt-in performance flags (off by default; named profiles turn them on).
    kv_ering: bool = False  # Alias KV DMA ring onto epool when unused
    pair_kv: bool = False  # Pair KV attention items to overlap KV DMA + softmax
    gqa_qv: bool = False  # Head-batched Q/K norm+RoPE and 128-lane Q/O staging
    gq_pack: bool = False  # Pack G*nh Q heads into 16-row MXU tiles (needs gqa_qv)
    rope_fp32: bool = False  # Keep RoPE intermediate products and sums in float32

    # `exact`: moving the field keeps hidden states and caches bit-identical
    # (scheduling only); `approx`: changes or may change the arithmetic.
    NUMERICS: ClassVar[dict[str, NumericsClass]] = {
        "kv_blk": "approx",  # online-softmax block size
        "nkv": "exact",  # KV ring depth (scheduling)
        "split": "approx",  # sequence-split attention (partial-softmax merge)
        "pack": "approx",  # split-attention row packing
        "kv_sub": "approx",  # online-softmax sub-block updates
        "amask": "approx",  # attention mask mode
        "kv_scatter_vec_ns": "exact",  # host KV update: one scatter vs per-sequence
        "kv_ering": "exact",  # KV ring staged in the expert pool (TPU bisect, CPU test)
        "pair_kv": "exact",  # paired KV items (TPU bisect, CPU test)
        "gqa_qv": "exact",  # head-batched Q/K norm + RoPE (TPU bisect, CPU test)
        "gq_pack": "approx",  # packed G*nh query tiles
        "rope_fp32": "approx",  # float32 vs bfloat16-rounded RoPE intermediates
    }


def derive_params(g: Any, ex: ExecSpec, plan: Any, tt: GQATunables) -> GQAParams:
    """`GQAParams` for geometry `g` under `ex` and the block's tunables `tt`."""
    tiles = ex.resolve_tiles(g)
    nh = g.NH // ex.R
    nai = (2 * nh + 2) * g.HD
    G = max(1, ex.R // g.NKV)
    BLK = tt.kv_blk
    PAIR_KV = tt.pair_kv
    NKV = max(tt.nkv, 4) if PAIR_KV else tt.nkv
    assert ex.ctx % BLK == 0 and BLK % 128 == 0, (ex.ctx, BLK)
    if cache_exec(ex).paged_kv:
        assert g.HD % 4 == 0, f"paged_kv requires HD % 4 == 0, got {g.HD}"
        assert (
            cache_exec(ex).page_size > 0
            and cache_exec(ex).page_size % 128 == 0
            and BLK % cache_exec(ex).page_size == 0
        ), (
            BLK,
            cache_exec(ex).page_size,
        )
        assert (
            ex.ctx % cache_exec(ex).page_size == 0
            and cache_exec(ex).num_pages >= 1
            and resolved_max_pages_per_seq(ex) >= ex.ctx // cache_exec(ex).page_size
        ), (
            cache_exec(ex).page_size,
            ex.ctx,
            cache_exec(ex).num_pages,
            resolved_max_pages_per_seq(ex),
        )
        assert tt.amask not in (3, 4), (
            f"paged_kv requires amask in {{0, 1, 2}}, got {tt.amask}"
        )
    KV_ITEM = 2 * (g.HD // 4) * BLK * 4
    EBROWS = tiles.ebank // 128
    # KVE stages KV blocks in the (idle) expert ring. On TPU this aliases the
    # untyped byte pool `epool` via `pool_alias.view`; in CPU interpret mode
    # `vmem.build_scratch` allocates a separate typed `epool`, so the
    # aliasing itself is only exercised on TPU (the KV math is shared).
    KVE = bool(
        cache_exec(ex).paged_kv
        and tt.kv_ering
        and bool(plan.dynamic_kinds)
        and tiles.nbe >= NKV
        and tiles.ebank >= KV_ITEM
        and g.HD == KVE_HD
    )
    NKVR = tiles.nbe if KVE else NKV
    SUB = tt.kv_sub or BLK
    assert BLK % SUB == 0 and SUB % 128 == 0, (BLK, SUB)
    SPLIT = tt.split and G > 1
    assert not SPLIT or (G & (G - 1) == 0), G
    TS = G if SPLIT else 1
    QR = rup(G * nh, 8) if SPLIT else 8
    TQ = ex.T * QR
    PW = g.HD + 128
    eff_ctx = (
        max(ex.ctx, resolved_max_pages_per_seq(ex) * cache_exec(ex).page_size)
        if cache_exec(ex).paged_kv
        else ex.ctx
    )
    max_nb = -(-eff_ctx // BLK)
    NWL = rup(ex.NS * -(-max_nb // TS), 8)
    PACK = (ex.WIDE if tt.pack < 0 else bool(tt.pack)) and SPLIT and nh < 8
    PB = 8 // nh if PACK else 1
    NBP = -(-ex.B // PB)
    NHH = g.HD // 128
    SR = 8 // PB
    QV = (
        tt.gqa_qv
        and (SPLIT or ex.VER)
        and g.HD % 128 == 0
        and nh <= 8
        and (not PACK or PB * nh == 8)
    )
    if tt.gq_pack and not tt.gqa_qv:
        raise MegakernelConfigError("GQATunables.gq_pack requires gqa_qv")
    QG = ex.T * nh
    GQ = bool(
        tt.gq_pack
        and QV
        and SPLIT
        and ex.VER
        and G * nh == QR
        and QG % 16 == 0
        and ex.NS <= 64
        and g.HD == GQ_ML_LANES
    )
    MLW = 128 // ex.NS if GQ else 0
    if GQ:
        SR = nh
    QVR = ex.B * nh + 8 if GQ else NBP * 8
    nt_att = plan.nt["att_in"] + plan.nt["att_out"]
    LAZY_ATT = tiles.lazy and nt_att <= tiles.nbd
    return GQAParams(
        H=g.H,
        nh=nh,
        HD=g.HD,
        ROT=g.ROT,
        NAI=nai,
        G=G,
        BLK=BLK,
        NKV=NKV,
        NKVR=NKVR,
        KVE=KVE,
        KV_ITEM=KV_ITEM,
        EBROWS=EBROWS,
        PAIR_KV=PAIR_KV,
        SUB=SUB,
        AMASK=tt.amask,
        SPLIT=SPLIT,
        TS=TS,
        QR=QR,
        TQ=TQ,
        PW=PW,
        NWL=NWL,
        PACK=PACK,
        PB=PB,
        NBP=NBP,
        QV=QV,
        GQ=GQ,
        NHH=NHH,
        SR=SR,
        QG=QG,
        MLW=MLW,
        QVR=QVR,
        LAZY_ATT=LAZY_ATT,
        eps=g.eps,
        theta=g.theta,
        zero_centered_norm=bool(g.zero_centered_norm),
        rope_fp32=bool(tt.rope_fp32),
    )


def gqa_scratch(ex: ExecSpec, ap: GQAParams) -> dict[str, Any]:
    """Allocate `GQABlock` VMEM/SMEM/semaphore scratch descriptors."""
    scr: dict[str, Any] = {}
    if not cache_exec(ex).paged_kv:
        scr["kbuf"] = pltpu.VMEM((ap.NKV, ap.BLK, ap.HD), BF16)
        scr["vbuf"] = pltpu.VMEM((ap.NKV, ap.BLK, ap.HD), BF16)
    scr.update(
        {
            "kv_sem": pltpu.SemaphoreType.DMA(
                (ap.NKVR,) if cache_exec(ex).paged_kv else (2, ap.NKV)
            ),
            "wl_b": pltpu.SMEM((ap.NWL,), I32),
            "wl_t": pltpu.SMEM((ap.NWL,), I32),
            "wl_cnt": pltpu.SMEM((1,), I32),
            "qb": pltpu.VMEM((ex.NS, ap.TQ, ap.HD), BF16),
            "am": pltpu.VMEM((ex.NS, ap.TQ, 128), F32),
            "al": pltpu.VMEM((ex.NS, ap.TQ, 128), F32),
            "aacc": pltpu.VMEM((ex.NS, ap.TQ, ap.HD), F32),
            "knew": pltpu.VMEM((ex.MR, ap.HD), F32),
            "vnew": pltpu.VMEM((ex.MR, ap.HD), F32),
            "gate": pltpu.VMEM((ex.MR, ap.nh * ap.HD), F32),
            "rcs": pltpu.VMEM((2, ex.MR, ap.ROT // 2), F32),
        }
    )
    if not cache_exec(ex).FUSED:
        scr["kv_out_sem"] = pltpu.SemaphoreType.DMA((2,))
    else:
        scr.update(fused_io().gqa_fused_scratch(ex, ap.HD))
    if cache_exec(ex).paged_kv:
        if not ap.KVE:
            scr["kvbuf"] = pltpu.VMEM((ap.NKV, 2, ap.HD // 4, 4, ap.BLK), FP8)
        scr["knew_o"] = pltpu.VMEM((ex.MR, ap.HD), FP8)
        scr["vnew_o"] = pltpu.VMEM((ex.MR, ap.HD), FP8)
        scr["wl_kp"] = pltpu.SMEM((ap.NWL * (ap.BLK // cache_exec(ex).page_size),), I32)
    if ap.SPLIT:
        q_rows = ap.NBP if ap.PACK else ex.B
        scr.update(
            {
                "qg": pltpu.VMEM((ap.G, q_rows, 8, ap.HD), F32),
                "pso": pltpu.VMEM((ap.G, q_rows, 8, ap.PW), F32),
                "pri": pltpu.VMEM((ap.G, q_rows, 8, ap.PW), F32),
                "sq_ssem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                "sq_rsem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                "sp_ssem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
                "sp_rsem": pltpu.SemaphoreType.DMA((ap.G - 1,)),
            }
        )
    elif ex.VER:
        scr["qg"] = pltpu.VMEM((1, ex.B, 8, ap.HD), F32)
    if ap.QV:
        scr["rope"] = pltpu.VMEM((2, ex.MR, 128), F32)  # QV-only cos / sin table
        scr["qg"] = pltpu.VMEM(
            (ap.G if ap.SPLIT and not ap.GQ else 1, ap.NHH, ap.QVR, 128), F32
        )
        scr["ost"] = pltpu.VMEM((ap.NHH, ap.QVR, 128), F32)
    if ap.GQ:
        del scr["pso"]
        scr.update(
            {
                "qxb": pltpu.VMEM((ap.G, ex.NS, ap.QG, ap.HD), BF16),
                "psend": pltpu.VMEM((ap.G, ex.NS + 1, ap.QG, ap.HD), F32),
                "pri": pltpu.VMEM((ap.G, ex.NS + 1, ap.QG, ap.HD), F32),
            }
        )
    return scr
