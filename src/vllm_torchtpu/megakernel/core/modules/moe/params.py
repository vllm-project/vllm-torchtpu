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
    # preconditions in `derive_moe_params`).
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
    # supersedes `ecopy_opt` whenever its static gate in `derive_moe_params`
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

    Each flag gates an optional kernel input: `forced_routing`
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

# `scr["seen"]` is `[8, 128]` int32 SMEM (zeroed from 8 VMEM rows of `hitv`).
MAX_SEEN_EXPERTS = 8 * 128

_ROUTE_VEC_FALLBACK_LOGGED: set[tuple[Any, ...]] = set()


def validate_moe_geometry(
    *,
    E: int,
    R: int,
    TOPK: int,
    IS: int,
    I: int,  # noqa: E741
    moe_tp: bool,
    e_slots: int,
) -> tuple[int, int, int, int]:
    """Validate MoE rank/expert/intermediate geometry and return `(er, isr, ir, ew)`."""
    if R <= 0:
        raise MegakernelConfigError(f"MoE R={R} must be >= 1")
    if E <= 0 or E % R != 0:
        raise MegakernelConfigError(f"MoE E={E} must be a positive multiple of R={R}")
    if TOPK <= 0 or min(E, 128) < TOPK:
        raise MegakernelConfigError(
            f"MoE TOPK={TOPK} must satisfy 1 <= TOPK <= min(E={E}, 128)"
        )
    if IS <= 0 or IS % R != 0:
        raise MegakernelConfigError(f"MoE IS={IS} must be a positive multiple of R={R}")
    if I <= 0 or (moe_tp and I % R != 0):
        raise MegakernelConfigError(
            f"MoE I={I} must be > 0"
            + (f" and divisible by R={R} when moe_tp=True" if moe_tp else "")
        )
    er = E // R
    isr = IS // R
    ir = I // R if moe_tp else I
    if e_slots < 0:
        raise MegakernelConfigError(f"e_slots={e_slots} must be >= 0")
    if moe_tp and e_slots not in (0, E):
        raise MegakernelConfigError(
            "moe_tp=True does not support redundant expert slots "
            f"(got e_slots={e_slots} for E={E})"
        )
    if not moe_tp and 0 < e_slots < er:
        raise MegakernelConfigError(
            f"e_slots={e_slots} must be 0 or >= ER={er} when moe_tp=False"
        )
    ew = e_slots if e_slots > 0 else (E if moe_tp else er)
    return er, isr, ir, ew


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
    if tt.topk and not (g.E > 0 and (g.E - 1).bit_length() <= 9):
        raise MegakernelConfigError(
            "topk_packed requires 1 <= E <= 512 ((E - 1).bit_length() <= 9 "
            f"for 9-bit float32 mantissa packing), got E={g.E}"
        )
    if tt.ecopy_opt not in (0, 1):
        raise MegakernelConfigError(
            f"MoETunables.ecopy_opt must be 0 or 1, got {tt.ecopy_opt!r}"
        )
    er, isr, ir, ew = validate_moe_geometry(
        E=g.E,
        R=ex.R,
        TOPK=g.TOPK,
        IS=g.IS,
        I=g.I,
        moe_tp=moe_tp,
        e_slots=e_slots,
    )
    extra = ew - er
    if g.E % 128 != 0:
        raise MegakernelConfigError(f"Dimension E={g.E} must be a multiple of 128")
    if g.E > MAX_SEEN_EXPERTS:
        raise MegakernelConfigError(
            f"MoE E={g.E} exceeds seen-table capacity {MAX_SEEN_EXPERTS}"
        )
    if g.H <= 0 or g.H % 128 != 0:
        raise MegakernelConfigError(f"Dimension H={g.H} must be a multiple of 128")
    if isr % 128 != 0:
        raise MegakernelConfigError(
            f"Dimension ISR={isr} (IS={g.IS}, R={ex.R}) must be a multiple of 128"
        )
    if ir % 128 != 0:
        raise MegakernelConfigError(
            f"Dimension IR={ir} (I={g.I}, R={ex.R}, moe_tp={moe_tp}) "
            "must be a multiple of 128"
        )
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
    nb_picks = round_up(ex.B * g.TOPK, 128)
    route_vec_checks = (
        ("ecopy", ecopy),  # redundant-expert placement table active
        ("E % 128 == 0", g.E % 128 == 0),  # 128-lane expert bitmask tiling
        ("(E >> 5) <= 256", (g.E >> 5) <= 256),  # bitmask rows fit 256-lane scratch
        ("E // 128 + 1 <= MR", g.E // 128 + 1 <= ex.MR),  # MROW + 1 <= MR scratch
        ("R * extra + R + 1 <= 128", ex.R * extra + ex.R + 1 <= 128),  # <= 128 lanes
        ("nb_picks >= ER", nb_picks >= er),  # elist/wlist covers ER local slots
    )
    failed_rvec = [name for name, ok in route_vec_checks if not ok]
    route_vec = bool(tt.route_vec and not failed_rvec)
    if tt.route_vec and failed_rvec:
        log_key = (g.E, ex.R, ex.B, ex.MR, ecopy, extra, nb_picks, er)
        if log_key not in _ROUTE_VEC_FALLBACK_LOGGED:
            _ROUTE_VEC_FALLBACK_LOGGED.add(log_key)
            logger.info(
                "MoETunables.route_vec=True disabled (failed: %s) for E=%d, R=%d, "
                "B=%d, MR=%d, ecopy=%s, extra=%d, nb_picks=%d, er=%d",
                ", ".join(failed_rvec),
                *log_key,
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
    )
