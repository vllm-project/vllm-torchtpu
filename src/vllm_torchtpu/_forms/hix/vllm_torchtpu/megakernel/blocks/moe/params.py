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
"""`MoEBlock` build-time parameters and tunables."""

from __future__ import annotations

import os  # q38 HIX700 PREGATE (the AR0S-T refusal reads the environment)
from dataclasses import dataclass
from typing import Any, ClassVar

from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec, NumericsClass
from ...ops import rup


@dataclass(frozen=True)
class MoEParams:
    """Block-owned build-time parameters for `MoEBlock`."""

    H: int
    E: int
    TOPK: int
    IR: int
    ISR: int
    ER: int
    EW: int
    EXTRA: int
    CH: int
    T13: int
    T2: int
    TE: int
    NBE: int
    NSS: int
    PIPE: bool
    PIPE_EP: bool
    LAZY_MOE: bool
    N_MOE: int
    ECOPY: bool
    moe_tp: bool
    stack_e: bool
    topk_packed: bool
    topk_lanes: bool
    # Resolved vectorized routing flag (`MoETunables.route_vec` gated by static
    # preconditions in `MoEBlock.plan_params`).
    route_vec: bool
    r2_direct_store: bool
    unroll: bool
    # Paired expert-ring releases + unrolled refill loop (unrolled / PIPE_EP
    # expert paths) for buckets up to `MoETunables.moe_group_rows` rows.
    group_rel: bool
    ecopy_opt: int
    eps: float
    zero_centered_norm: bool
    # q38 HIX700 PREGATE: predicted routed experts per rank prefetched into the idle expert ring before routing (0 = off)
    # and where the prefetch is issued ("ar": before the mixer sublayer's all-reduce; "start": at layer start).
    PG: int = 0
    PG_AT: str = "ar"
    # "list" (the pg_pred debug input) or "pred" (in-kernel predictor: the mixer's HC-mixed input x this rank's
    # router columns, read as one 128-lane window of the router's HBM tiles); router tiling for the window DMA.
    PG_SRC: str = "list"
    PG_RNT: int = 0
    PG_RTK: int = 0


@dataclass(frozen=True)
class MoETunables:
    """`MoEBlock` knobs (`Tiles.blocks["moe"]`); `NUMERICS` classes each field."""

    pipe: bool = True  # Software-pipeline single-tile experts
    topk: bool = True  # One-reduction top-k picks on packed (value, lane) keys
    ecopy_opt: int = 0  # Redundant expert dedup optimization (0 or 1)
    # Unrolled / PIPE_EP experts: paired ring releases + unrolled refills up to
    # this many rows
    moe_group_rows: int = 40
    # Opt-in performance flags (off by default; named profiles turn them on).
    pipe_ep: bool = False  # Software-pipeline EP experts with T13 == 2 * T2
    r2_direct_store: bool = False  # Direct VMEM store on r2 tile 0
    # Vectorized MXU expert route packing (a distinct routing/Po2C policy that
    # supersedes `ecopy_opt` whenever its static gate in `MoEBlock.plan_params`
    # holds; resolved into `MoEParams.route_vec`)
    route_vec: bool = False
    topk_lanes: bool = False  # Batcher odd-even top-k on lanes for E >= 128
    moe_unroll: bool = False  # Unrolled expert tile loops + grouped expert refills
    # q38 HIX700 PREGATE: prefetch this many predicted routed experts per rank into the expert ring before routing; the
    # prefetched experts that routing hits run first into side buffers and are added back in hit-list order (bitwise).
    pregate: int = 0
    pregate_at: str = "ar"  # "ar" (before the mixer all-reduce) or "start" (layer start: the contention arm)

    # `exact`: moving the field keeps hidden states and caches bit-identical
    # (scheduling only); `approx`: changes or may change the arithmetic.
    NUMERICS: ClassVar[dict[str, NumericsClass]] = {
        "pipe": "approx",  # single-tile expert pipelining
        "topk": "approx",  # packed-key top-k
        "ecopy_opt": "approx",  # redundant-expert dispatch policy
        "moe_group_rows": "exact",  # paired expert-ring releases (scheduling)
        "pipe_ep": "approx",  # EP expert software pipeline
        "r2_direct_store": "approx",  # `=` instead of zero-fill + `+=` (signed zero)
        "route_vec": "approx",  # vectorized routing policy
        "topk_lanes": "approx",  # Batcher odd-even lane top-k
        "moe_unroll": "approx",  # unrolled expert tiles (not exact on TPU, bisect)
        "pregate": "exact",  # q38 HIX700 PREGATE: DMA order + side-buffer replay in hit-list order
        "pregate_at": "exact",  # q38 HIX700 PREGATE: DMA issue point only
    }


@dataclass(frozen=True)
class MoEDebug:
    """`MoEBlock` debug flags (`DebugOptions.blocks["moe"]`), not tuning knobs.

    Each flag adds a kernel input (`MoEBlock.extra_inputs`): `forced_routing`
    routes from an external `rbias` HBM input `[L, M, E]` per rank instead of
    the router's probabilities (the expert weights still come from the
    router); `row_mask` builds the routed-expert list only from the rows whose
    `moe_rows` SMEM entry (`[B]` int32) is non-zero (every row still adds its
    weighted output of the experts in that list).

    `hit_trace` (q38 X-K-LIVE, L3-LIVE; tests only) adds no input but three per-layer outputs that copy what the
    routing decided (`blocks/moe/trace.py`): the rows' picks (`hitv`), the hit list (`elist`) and the expert-unit
    count (`st` = n_hit * TE, the units `route_fetch` and the refills issue as expert-weight DMAs).
    """

    forced_routing: bool = False
    row_mask: bool = False
    hit_trace: bool = False
    # q38 HIX700 PREGATE (tests / microbench): the predicted experts come from an external `pg_pred` HBM input
    # `[L, R, 128]` int32 (global expert ids, -1 = empty slot) instead of the in-kernel predictor.
    pregate_list: bool = False
    # q38 HIX700 PREGATE (tests only): copy the MoE block's f32 output `part` (shared + routed, before the all-reduce)
    # of every layer to a `moe_trace_part` output, so a bitwise check sees the f32 summation order the BF16
    # all-reduce payload would mask.
    part_trace: bool = False
    # q38 HIX700 PREGATE (tests only): copy the predictor's input (the mixer's HC-mixed `hc_xb`) of every layer to a
    # `moe_trace_pgx` output, so a host reference can recompute the predicted experts.
    pregate_xtrace: bool = False


def derive_moe_params(
    g: Any,
    ex: ExecSpec,
    plan: Any,
    tt: MoETunables,
    dbg: MoEDebug,
    *,
    moe_tp: bool,
    e_slots: int,
) -> MoEParams:
    """Resolve `MoEParams` from model geometry, `ExecSpec`, and `StreamPlan`."""
    tiles = ex.resolve_tiles(g)
    er = g.E // ex.R
    isr = g.IS // ex.R
    ir = g.I // ex.R if moe_tp else g.I
    ew = e_slots if e_slots > 0 else (g.E if moe_tp else er)
    extra = ew - er
    ch = min(1024, g.H)
    assert g.H % ch == 0, (g.H, ch)
    t13, t2 = plan.nt["e13"], plan.nt["e2"]
    te = t13 + t2
    nbe = tiles.nbe
    nss = 3 + (nbe - 1) // te
    pipe = tt.pipe and t13 == 1 and t2 == 1
    pipe_ep = tt.pipe_ep and not pipe and t13 == 2 * t2 and nbe >= te and nbe % te == 0
    n_moe = plan.nt["router"] + plan.nt["sh13"] + plan.nt["sh2"]
    lazy_moe = tiles.lazy and n_moe <= tiles.nbd
    ecopy = not moe_tp and e_slots > er
    assert not (dbg.row_mask and ecopy), "row_mask: identity expert placement only"
    if ex.live_rows and ecopy:  # q38 X-K-LIVE (L3-LIVE): exclusive with X-K-ECOPY until composed
        raise MegakernelConfigError(
            "live_rows (X-K-LIVE) with redundant expert copies (ECOPY, e_slots > E / R) is not composed yet"
        )
    if ex.live_rows and dbg.row_mask:  # one row-mask source per kernel
        raise MegakernelConfigError("live_rows and the MoEDebug.row_mask debug input are exclusive")
    if tt.topk and not (g.E > 0 and (g.E - 1).bit_length() <= 9):
        raise MegakernelConfigError(
            "topk_packed requires 1 <= E <= 512 ((E - 1).bit_length() <= 9 "
            f"for 9-bit float32 mantissa packing), got E={g.E}"
        )
    nb_picks = rup(ex.B * g.TOPK, 128)
    route_vec = bool(
        tt.route_vec
        and ecopy  # redundant-expert placement table active
        and g.E > 0
        and g.E % 128 == 0  # 128-lane expert bitmask tiling
        and (g.E >> 5) <= 256  # packed bitmask rows fit 256-lane scratch
        and g.E // 128 + 1 <= ex.MR  # MROW + 1 <= MR for route_topk scratch
        and ex.R * extra + ex.R + 1 <= 128  # NCP + R + 1 <= 128 VREG lanes
        and nb_picks >= er  # elist/wlist capacity covers ER local slots
    )
    return MoEParams(
        H=g.H,
        E=g.E,
        TOPK=g.TOPK,
        IR=ir,
        ISR=isr,
        ER=er,
        EW=ew,
        EXTRA=extra,
        CH=ch,
        T13=t13,
        T2=t2,
        TE=te,
        NBE=nbe,
        NSS=nss,
        PIPE=pipe,
        PIPE_EP=pipe_ep,
        LAZY_MOE=lazy_moe,
        N_MOE=n_moe,
        ECOPY=ecopy,
        moe_tp=moe_tp,
        stack_e=ex.stack_e,
        topk_packed=tt.topk,
        topk_lanes=tt.topk_lanes,
        route_vec=route_vec,
        r2_direct_store=tt.r2_direct_store,
        unroll=bool(tt.moe_unroll or pipe_ep),
        group_rel=tt.moe_group_rows >= ex.B,
        ecopy_opt=tt.ecopy_opt,
        eps=g.eps,
        zero_centered_norm=bool(getattr(g, "zero_centered_norm", True)),
        **_pregate_params(tt, ex, dbg, g, plan, moe_tp=moe_tp, ecopy=ecopy, pipe=pipe, pipe_ep=pipe_ep, te=te, nbe=nbe, nss=nss),
    )


# q38 HIX700 PREGATE begin. Appended so the flag-off lowering of every function above keeps its source locations.
# q38 HIX700 PREGATE x HXSMALL AR0S-T (HIX-LEAD-6 ruling D-2, 2026-11-23): S2's AR0S-T (VLLM_TPU_HXS_AR0ST=1) is a branch of
# moe/block.py's expert-path choice that PREGATE's branch precedes, so with both armed AR0S-T would be skipped with no line.
# Until a composed path exists PREGATE refuses to build with it (fail closed; read from the environment, so the refusal
# holds on trees without HXSMALL's key registry). Called by every kernel build with pregate > 0 (`_pregate_params`) and by
# the served construction (megakernel_runtime/hix700_pregate.py `pregate_tiles`, warm-cache safe).
PG_REFUSE_AR0ST = "q38 HIX700 PREGATE refuses AR0S-T (no composed path)"


def pregate_refuse_ar0st(pg: int) -> None:
    v = os.environ.get("VLLM_TPU_HXS_AR0ST", "0").strip()
    if pg > 0 and v not in ("", "0"):
        raise MegakernelConfigError(f"{PG_REFUSE_AR0ST}: VLLM_TPU_HXS_AR0ST={v!r} with pregate={pg}; arm one of the two")


def _pregate_params(tt: MoETunables, ex: ExecSpec, dbg: MoEDebug, g: Any, plan: Any, *, moe_tp: bool, ecopy: bool,
                    pipe: bool, pipe_ep: bool, te: int, nbe: int, nss: int) -> dict:
    """`PG` / `PG_AT` for `MoEParams`; refuses the paths PREGATE does not cover (fail closed, never silently off)."""
    pg = int(tt.pregate)
    if pg <= 0:
        return {}
    pregate_refuse_ar0st(pg)
    at = str(tt.pregate_at)
    if at not in ("ar", "start"):
        raise MegakernelConfigError(f"pregate_at must be 'ar' or 'start', got {at!r}")
    why = [n for n, bad in (("moe_tp", moe_tp), ("ECOPY", ecopy), ("PIPE", pipe), ("PIPE_EP", pipe_ep),
                            ("stack_e", bool(ex.stack_e))) if bad]
    if why:
        raise MegakernelConfigError(f"pregate (HIX700 PREGATE) covers the EP expert() path only; refused with {why}")
    if pg * te > nbe or pg > 8 or pg > nss - 1:
        raise MegakernelConfigError(f"pregate={pg}: needs pregate * TE <= NBE ({te} * {pg} vs {nbe}), <= 8 and < NSS {nss}")
    src = "list" if dbg.pregate_list else "pred"
    er = g.E // ex.R
    if src == "pred":
        if at != "ar":
            raise MegakernelConfigError("pregate's in-kernel predictor runs at the mixer (pregate_at='ar' only)")
        if not (er <= 128 and 128 % er == 0) or g.E % 128:
            raise MegakernelConfigError(f"pregate predictor: E / R = {er} must divide 128 (one 128-lane router window)")
    if dbg.hit_trace or "experts" in ex.skip:  # (part_trace composes: it only copies `part` after the experts)
        raise MegakernelConfigError("pregate is not composed with MoEDebug.hit_trace or skip=experts")
    if pg + min(g.E // ex.R, ex.B * g.TOPK) > rup(ex.B * g.TOPK, 128):
        raise MegakernelConfigError("pregate: the combined hit list does not fit the elist scratch")
    return {"PG": pg, "PG_AT": at, "PG_SRC": src, "PG_RNT": int(plan.nt["router"]), "PG_RTK": int(plan.tk["router"])}
# q38 HIX700 PREGATE end

