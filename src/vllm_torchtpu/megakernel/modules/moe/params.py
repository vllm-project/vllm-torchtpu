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

from dataclasses import dataclass
from typing import Any, ClassVar

from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec, NumericsClass
from ...ops import col_chunk, rup


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
    assert not (dbg.row_mask and ecopy), "row_mask: identity expert placement only"
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
    )
