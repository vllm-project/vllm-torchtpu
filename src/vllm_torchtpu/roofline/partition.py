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
"""Pipeline-parallel layer counting and dynamic-programming stage partitioning."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .constants import NON_FULL_ATTN_TOKENS
from .geometry import layer_ffn_kinds
from .memory import HbmBreakdown, SeqKvLayout, hbm_breakdown_per_core_bytes

if TYPE_CHECKING:
    from .geometry import ModelGeometry
    from .modules import Dtypes, Parallel


@dataclass(frozen=True, slots=True)
class PpStageOptions:
    """Options controlling disaggregated stage feasibility and max layer count."""

    disagg: bool = False
    max_stage_layers: int | None = None


@dataclass(frozen=True, slots=True)
class _PpLayerPrefixes:
    """Prefix sums of per-layer memory footprints and full-attention counts."""

    weights: list[float]
    kv: list[float]
    state: list[float]
    full: list[int]
    embed_b: float
    lm_b: float


@dataclass(frozen=True, slots=True)
class _PpDpProblem:
    """DP state bounds and prefix sums for contiguous pipeline partitioning."""

    pp: int
    n_l: int
    max_stage_layers: int | None
    min_full: int
    pref: _PpLayerPrefixes


@dataclass(frozen=True, slots=True)
class _DpTables:
    """Dynamic programming cost and backpointer tables."""

    cost: list[list[float]]
    prev: list[list[int]]


def count_full_attn_layers(geom: ModelGeometry) -> int:
    """Count standard full-attention layers excluding linear and sliding variants."""
    return sum(1 for tok in geom.layer_types if tok not in NON_FULL_ATTN_TOKENS)


def _pp_layer_prefixes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par1: Parallel,
    layout: SeqKvLayout,
) -> _PpLayerPrefixes:
    """Compute prefix sums of weights, KV, state, and full-attention counts."""
    n_l = geom.num_layers
    tp = max(1, par1.tp)
    w_embed = float(geom.vocab * geom.hidden) * dtypes.lm_head / float(tp)
    if geom.mtp_layers > 0:
        bd_with_mtp = hbm_breakdown_per_core_bytes(
            geom, dtypes, par1, SeqKvLayout(0, 0)
        )
        bd_no_mtp = hbm_breakdown_per_core_bytes(
            replace(geom, mtp_layers=0), dtypes, par1, SeqKvLayout(0, 0)
        )
        w_mtp_total = max(0.0, bd_with_mtp.weights - bd_no_mtp.weights)
    else:
        w_mtp_total = 0.0
    w_lm = w_embed + w_mtp_total
    w_head_both = 2.0 * w_embed

    layer_w: list[float] = []
    layer_kv: list[float] = []
    layer_st: list[float] = []
    is_full: list[int] = []
    ffn_kinds = layer_ffn_kinds(geom)
    for idx in range(n_l):
        cr_i = (geom.compress_ratios[idx],) if idx < len(geom.compress_ratios) else ()
        f_kind = ffn_kinds[idx] if idx < len(ffn_kinds) else "dense"
        g_i = replace(
            geom,
            num_layers=1,
            layer_types=(geom.layer_types[idx],),
            compress_ratios=cr_i,
            ffn_kinds=(f_kind,),
            n_moe_layers=1 if f_kind == "moe" else 0,
            n_dense_layers=1 if f_kind == "dense" else 0,
            first_k_dense_replace=0 if f_kind == "moe" else 1,
            mtp_layers=0,
        )
        bd_i = hbm_breakdown_per_core_bytes(g_i, dtypes, par1, layout)
        layer_w.append(max(0.0, bd_i.weights - w_head_both))
        layer_kv.append(bd_i.kv)
        layer_st.append(bd_i.state)
        is_full.append(0 if geom.layer_types[idx] in NON_FULL_ATTN_TOKENS else 1)

    pref_w = [0.0] * (n_l + 1)
    pref_kv = [0.0] * (n_l + 1)
    pref_st = [0.0] * (n_l + 1)
    pref_full = [0] * (n_l + 1)
    for i in range(n_l):
        pref_w[i + 1] = pref_w[i] + layer_w[i]
        pref_kv[i + 1] = pref_kv[i] + layer_kv[i]
        pref_st[i + 1] = pref_st[i] + layer_st[i]
        pref_full[i + 1] = pref_full[i] + is_full[i]
    return _PpLayerPrefixes(
        weights=pref_w,
        kv=pref_kv,
        state=pref_st,
        full=pref_full,
        embed_b=w_embed,
        lm_b=w_lm,
    )


def _check_uniform_pp(
    unif: tuple[int, ...],
    pref: _PpLayerPrefixes,
    min_full: int,
    target_cost: float,
) -> bool:
    """Check if uniform layer partition is legal and cost-competitive."""
    pp = len(unif)
    u_max = 0.0
    pos = 0
    for s, cnt in enumerate(unif):
        nxt = pos + cnt
        if min_full > 0 and (pref.full[nxt] - pref.full[pos]) < min_full:
            return False
        extra_stage_bytes = (pref.embed_b if s == 0 else 0.0) + (
            pref.lm_b if s == pp - 1 else 0.0
        )
        u_max = max(
            u_max,
            (pref.weights[nxt] - pref.weights[pos])
            + (pref.kv[nxt] - pref.kv[pos])
            + (pref.state[nxt] - pref.state[pos])
            + extra_stage_bytes,
        )
        pos = nxt
    return u_max <= target_cost * (1.0 + 1e-6)


def _relax_stage_dp(
    stage_idx: int,
    extra: float,
    prob: _PpDpProblem,
    dp: _DpTables,
) -> None:
    """Relax DP transitions for one pipeline stage index."""
    inf = math.inf
    pref = prob.pref
    rem_stages = prob.pp - stage_idx
    for b in range(stage_idx, prob.n_l - rem_stages + 1):
        best_c = inf
        best_a = -1
        a_min = (
            max(stage_idx - 1, b - prob.max_stage_layers)
            if prob.max_stage_layers is not None
            else (stage_idx - 1)
        )
        for a in range(a_min, b):
            prev_cost = dp.cost[stage_idx - 1][a]
            if prev_cost == inf:
                continue
            if prob.min_full > 0 and (pref.full[b] - pref.full[a]) < prob.min_full:
                continue
            seg = (
                (pref.weights[b] - pref.weights[a])
                + (pref.kv[b] - pref.kv[a])
                + (pref.state[b] - pref.state[a])
                + extra
            )
            cand = max(prev_cost, seg)
            if cand < best_c:
                best_c = cand
                best_a = a
        dp.cost[stage_idx][b] = best_c
        dp.prev[stage_idx][b] = best_a


def _dp_solve_pp(prob: _PpDpProblem) -> tuple[tuple[int, ...], bool]:
    """Solve DP partitioning for contiguous pipeline stages."""
    inf = math.inf
    dp = _DpTables(
        cost=[[inf] * (prob.n_l + 1) for _ in range(prob.pp + 1)],
        prev=[[-1] * (prob.n_l + 1) for _ in range(prob.pp + 1)],
    )
    dp.cost[0][0] = 0.0

    for s in range(1, prob.pp + 1):
        extra = (prob.pref.embed_b if s == 1 else 0.0) + (
            prob.pref.lm_b if s == prob.pp else 0.0
        )
        _relax_stage_dp(s, extra, prob, dp)

    if dp.prev[prob.pp][prob.n_l] < 0:
        base, rem = prob.n_l // prob.pp, prob.n_l % prob.pp
        return (tuple(base + (1 if s < rem else 0) for s in range(prob.pp)), False)

    rev: list[int] = []
    cur = prob.n_l
    for s in range(prob.pp, 0, -1):
        prv = dp.prev[s][cur]
        rev.append(cur - prv)
        cur = prv
    part = tuple(reversed(rev))

    if prob.n_l % prob.pp == 0:
        unif = (prob.n_l // prob.pp,) * prob.pp
        if _check_uniform_pp(
            unif, prob.pref, prob.min_full, dp.cost[prob.pp][prob.n_l]
        ):
            part = unif
    return (part, True)


def _pp_part_breakdown(
    part: tuple[int, ...],
    pref: _PpLayerPrefixes,
    pp: int,
) -> tuple[HbmBreakdown, float]:
    """Calculate the peak stage HBM breakdown and last-stage footprint."""
    best_bd = HbmBreakdown(0.0, 0.0, 0.0, -1.0)
    last_tot = 0.0
    pos = 0
    for s, cnt in enumerate(part):
        nxt = pos + cnt
        ex_w = (pref.embed_b if s == 0 else 0.0) + (pref.lm_b if s == pp - 1 else 0.0)
        sw = (pref.weights[nxt] - pref.weights[pos]) + ex_w
        skv = pref.kv[nxt] - pref.kv[pos]
        sst = pref.state[nxt] - pref.state[pos]
        stot = sw + skv + sst
        if s == pp - 1:
            last_tot = stot
        if stot > best_bd.total:
            best_bd = HbmBreakdown(weights=sw, kv=skv, state=sst, total=stot)
        pos = nxt
    return (best_bd, last_tot)


def pp_stage_breakdowns(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    workload: tuple[int, int] = (0, 1),
    options: PpStageOptions | None = None,
) -> tuple[tuple[int, ...], HbmBreakdown, float, bool]:
    """Dynamic-programming contiguous PP layer partition and HBM breakdown."""
    context, seqs_per_dp = workload
    opts = options or PpStageOptions()
    pp = max(1, parallel.pp)
    n_l = geom.num_layers
    layout = SeqKvLayout(seqs_per_dp=seqs_per_dp, context=context)
    if pp <= 1 or n_l <= 0:
        bd = hbm_breakdown_per_core_bytes(geom, dtypes, parallel, layout)
        return ((n_l,), bd, bd.total, True)
    par1 = replace(parallel, pp=1)
    pref = _pp_layer_prefixes(geom, dtypes, par1, layout)
    # Require every stage to retain at least one full-attention layer in disagg hybrid
    # serving regardless of whether pp <= n_full (raises infeasible when pp > n_full).
    req_full = opts.disagg and 0 < pref.full[n_l] < n_l
    prob = _PpDpProblem(
        pp=pp,
        n_l=n_l,
        max_stage_layers=opts.max_stage_layers,
        min_full=1 if req_full else 0,
        pref=pref,
    )
    part, feasible = _dp_solve_pp(prob)
    best_bd, last_tot = _pp_part_breakdown(part, pref, pp)
    return (part, best_bd, last_tot, feasible)


def partition_pp_layers(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    workload: tuple[int, int] = (0, 1),
    *,
    disagg: bool = False,
) -> tuple[int, ...]:
    """Dynamic-programming contiguous PP layer partition minimizing max-stage HBM."""
    part, _, _, feasible = pp_stage_breakdowns(
        geom,
        dtypes,
        parallel,
        workload=workload,
        options=PpStageOptions(disagg=disagg),
    )
    if disagg and not feasible:
        msg = (
            f"Cannot partition {geom.num_layers} layers into {max(1, parallel.pp)} "
            "disagg pipeline stages with a full-attention layer per stage"
        )
        raise ValueError(msg)
    return part
