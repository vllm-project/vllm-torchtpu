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
"""`MoEModule` build-time parameters and tunables."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec, NumericsClass, register_debug, register_tunables
from ...ops import col_chunk, round_up

logger = logging.getLogger(__name__)

# `route_shrink`: bit position of the copy-slot index packed above the expert id
# in a copy-list entry.
CP_SH = 16


@dataclass(frozen=True)
class MoEParams:
    """Module-owned build-time parameters for `MoEModule`."""

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
    # preconditions in `MoEModule.plan_params`).
    route_vec: bool
    r2_direct_store: bool
    unroll: bool
    # Paired expert-ring releases + unrolled refill loop (unrolled / PIPE_EP
    # expert paths) for buckets up to `MoETunables.moe_group_rows` rows.
    group_rel: bool
    ecopy_opt: int
    eps: float
    zero_centered_norm: bool
    sh13_scaled: bool = True
    sh2_scaled: bool = True
    # `MoETunables.ep_fast_refill` resolved: PIPE_EP with one whole expert per
    # ring (NBE == TE) and unstacked expert weights.
    ep_fast: bool = False
    # `MoETunables.route_early`: routing weights under the hit-list copy, and
    # (vectorized routing) the first expert fills right after the fixed-hit scan.
    route_early: bool = False
    # `MoETunables.cp_interleave` / `route_shrink` resolved (vectorized routing
    # with redundant copies; `route_shrink` also needs E < 2**CP_SH).
    cp_interleave: bool = False
    route_shrink: bool = False
    # `MoETunables.prim_now` resolved (with `route_shrink`, ER dividing 128): the
    # fixed-hit scan stores no `wlist` entries; slot = expert - me * ER for hit
    # list entries below `st[5]` (the fixed-hit count).
    prim_now: bool = False
    # `MoETunables.sh_pre` resolved (lazy MoE segment, shared expert of at most
    # 4 dense tiles, each kind within `dmm_unroll`): the shared expert's tiles
    # are taken with the router's and its matmuls run from registers before
    # (`sh_at` 0), inside (1) or after (2) routing. `sh_hoist`: the router +
    # shared tiles wait as one hoisted group (`Tiles.dwait_hoist`, router
    # fully unrolled).
    sh_pre: bool = False
    sh_hoist: bool = False
    sh2_pre_route: bool = False
    sh_at: int = 0
    # `MoETunables.flush_pre_drain` resolved (lazy MoE segment, PIPE_EP).
    flush_pre_drain: bool = False
    # `MoETunables.flush_in_ar1` resolved (lazy MoE segment, not PIPE_EP): the
    # segment's dense refill is handed to the builder (`ctx.deferred_fill`),
    # which issues it inside the MoE all-reduce's first cross-chip round.
    flush_in_ar1: bool = False


@dataclass(frozen=True)
class MoETunables:
    """`MoEModule` knobs (`Tiles.modules["moe"]`); `NUMERICS` classes each field."""

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
    # supersedes `ecopy_opt` whenever its static gate in `MoEModule.plan_params`
    # holds; resolved into `MoEParams.route_vec`)
    route_vec: bool = False
    topk_lanes: bool = False  # Batcher odd-even top-k on lanes for E >= 128
    moe_unroll: bool = False  # Unrolled expert tile loops + grouped expert refills
    # Routing / shared-expert / refill schedule options (off by default).
    # Copy slot packed above the expert id in the copy list (shorter scalar chain)
    route_shrink: bool = False
    prim_now: bool = False  # Fixed-hit compaction stores no worklist entries
    sh_pre: bool = False  # Shared expert before routing (short shared-expert streams)
    sh2_pre_route: bool = False  # With `sh_pre`: its down projection before routing
    flush_in_ar1: bool = False  # Dense-ring flush inside the following all-reduce
    sh_at: int = 0  # `sh_pre` placement: 0 before route, 1 inside, 2 after first DMAs
    ep_fast_refill: bool = False  # EP expert refills with static banks
    # Vectorized routing: scan the copy slots rank-minor (slot j holds rank
    # j % R's extra slot j // R) instead of rank-major (`route_shrink` needs it)
    cp_interleave: bool = False
    # Routing weights under the hit-list copy; the first expert fills right
    # after the fixed-hit scan (vectorized routing)
    route_early: bool = False
    # Pipelined EP experts: the lazy dense-ring refill just before the last
    # expert's down projection instead of after the expert loop
    flush_pre_drain: bool = False

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
        # `approx` until a bit-identity flip (`EXACT_FLIPS`) shows otherwise.
        "route_shrink": "exact",  # same Po2C decisions (packed decode)
        "prim_now": "exact",  # derived instead of stored slots
        "sh_pre": "exact",  # same tile sums, moved before routing
        "sh2_pre_route": "exact",  # same ops, moved
        "flush_in_ar1": "exact",  # same refill, inside the all-reduce
        "sh_at": "exact",  # same ops, moved
        "ep_fast_refill": "exact",  # same DMAs, static banks
        "cp_interleave": "approx",  # copy-slot scan order (Po2C is greedy)
        "route_early": "exact",  # same ops and DMAs, earlier
        "flush_pre_drain": "exact",  # same refill, earlier
    }


@dataclass(frozen=True)
class MoEDebug:
    """`MoEModule` debug flags (`DebugOptions.modules["moe"]`), not tuning knobs.

    Each flag adds a kernel input (`MoEModule.extra_inputs`): `forced_routing`
    routes from an external `rbias` HBM input `[L, M, E]` per rank instead of
    the router's probabilities (the expert weights still come from the
    router); `row_mask` builds the routed-expert list only from the rows whose
    `moe_rows` SMEM entry (`[B]` int32) is non-zero (every row still adds its
    weighted output of the experts in that list).
    """

    forced_routing: bool = False
    row_mask: bool = False


register_tunables("moe", MoETunables)
register_debug("moe", MoEDebug)


def derive_moe_params(
    g: Any,
    ex: ExecSpec,
    plan: Any,
    tt: MoETunables,
    dbg: MoEDebug,
    *,
    moe_tp: bool,
    e_slots: int,
    module_name: str = "moe",
    sh13_scaled: bool = True,
    sh2_scaled: bool = True,
) -> MoEParams:
    """Resolve `MoEParams` from model geometry, `ExecSpec`, and `StreamPlan`."""
    tiles = ex.resolve_tiles(g)
    moe_layers = plan.module_layer_counts.get(module_name, g.L)
    if moe_layers != g.L:
        raise MegakernelConfigError(
            f"MoEModule requires execution on every layer (got {moe_layers} "
            f"of L={g.L} layers)"
        )
    er = g.E // ex.R
    isr = g.IS // ex.R
    ir = g.I // ex.R if moe_tp else g.I
    if e_slots < 0:
        raise MegakernelConfigError(f"e_slots={e_slots} must be >= 0")
    if moe_tp and e_slots not in (0, g.E):
        raise MegakernelConfigError(
            "moe_tp=True does not support redundant expert slots "
            f"(got e_slots={e_slots} for E={g.E})"
        )
    if not moe_tp and 0 < e_slots < er:
        raise MegakernelConfigError(
            f"e_slots={e_slots} must be 0 or >= ER={er} when moe_tp=False"
        )
    ew = e_slots if e_slots > 0 else (g.E if moe_tp else er)
    extra = ew - er
    if g.H <= 0 or g.H % 128 != 0:
        raise MegakernelConfigError(f"Dimension H={g.H} must be a multiple of 128")
    ch = col_chunk(g.H)
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
    if dbg.row_mask and ecopy:
        raise MegakernelConfigError(
            "MoEDebug.row_mask requires identity expert placement "
            f"(got ecopy=True with moe_tp=False, e_slots={e_slots} > ER={er})"
        )
    if tt.topk and not (g.E > 0 and (g.E - 1).bit_length() <= 9):
        raise MegakernelConfigError(
            "topk_packed requires 1 <= E <= 512 ((E - 1).bit_length() <= 9 "
            f"for 9-bit float32 mantissa packing), got E={g.E}"
        )
    nb_picks = round_up(ex.B * g.TOPK, 128)
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
    if tt.sh_at not in (0, 1, 2):
        raise MegakernelConfigError(
            f"MoETunables.sh_at must be 0, 1 or 2, got {tt.sh_at!r}"
        )
    # sh_pre sums the shared expert's tiles in registers, which is the dense
    # matmul's own order only on its fully unrolled path (nt <= dmm_unroll).
    dmu = min(tiles.dmm_unroll, tiles.nbd)
    sh_pre = bool(
        tt.sh_pre
        and lazy_moe
        and plan.nt["sh13"] + plan.nt["sh2"] <= 4
        and max(plan.nt["sh13"], plan.nt["sh2"]) <= dmu
    )
    if tt.route_shrink and not tt.cp_interleave:
        raise MegakernelConfigError(
            "MoETunables.route_shrink decodes the copy slot as rank j % R, slot "
            "j // R, so it requires cp_interleave"
        )
    if tt.route_vec and not route_vec:
        logger.debug(
            "MoETunables.route_vec=True disabled for E=%d, R=%d, B=%d, MR=%d, "
            "ecopy=%s, extra=%d, nb_picks=%d, er=%d",
            g.E,
            ex.R,
            ex.B,
            ex.MR,
            ecopy,
            extra,
            nb_picks,
            er,
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
        sh13_scaled=bool(sh13_scaled),
        sh2_scaled=bool(sh2_scaled),
        ep_fast=bool(tt.ep_fast_refill and pipe_ep and nbe == te and not ex.stack_e),
        route_early=bool(tt.route_early),
        cp_interleave=bool(tt.cp_interleave and route_vec),
        route_shrink=bool(tt.route_shrink and route_vec and g.E < (1 << CP_SH)),
        prim_now=bool(
            tt.prim_now
            and tt.route_shrink
            and route_vec
            and g.E < (1 << CP_SH)
            and 128 % er == 0
        ),
        sh_pre=sh_pre,
        sh_hoist=bool(sh_pre and tiles.dwait_hoist and plan.nt["router"] <= dmu),
        sh2_pre_route=bool(sh_pre and tt.sh2_pre_route),
        sh_at=int(tt.sh_at) if sh_pre else 0,
        flush_pre_drain=bool(tt.flush_pre_drain and lazy_moe and pipe_ep),
        flush_in_ar1=bool(tt.flush_in_ar1 and lazy_moe and not pipe_ep),
    )
