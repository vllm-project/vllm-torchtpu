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
"""Pipeline-parallel layer partitioning, stage KV sizing, and stage step floors."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal, TypedDict, Unpack

from .collectives import CollectiveConfig, make
from .constants import ACTIVATION_BYTES_BF16
from .geometry import HostTableResidency, layer_ffn_kinds, slice_geometry_layers
from .memory import (
    HbmBreakdown,
    SeqKvLayout,
    _compute_w_head,
    hbm_breakdown_per_core_bytes,
    kv_bytes_per_core_parts,
)
from .modules import ModuleCost, pass_costs
from .step import (
    FloorOptions,
    StepFloor,
    StepShape,
    combine_modules,
    prefill_draft_modules,
    step_floor,
)

if TYPE_CHECKING:
    from .chips import ChipSpec
    from .geometry import ModelGeometry
    from .modules import Dtypes, Parallel

_UNIFORM_TIE_TOL: Final[float] = 1e-6
_PP_PREFIX_CACHE_MAXSIZE: Final[int] = 4096
_PP_STAGE_FLOOR_CACHE_MAXSIZE: Final[int] = 8192


def _pp_handoff_mods(
    geometry: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    shape: StepShape,
) -> list[ModuleCost]:
    """Construct pipeline P2P handoff collective module directly from step shape."""
    if parallel.pp <= 1:
        return []
    decode_rows = shape.batch * shape.rows_per_seq
    raw_uncached = (
        shape.prefill_tokens
        * max(0.0, 1.0 - shape.prefix_hit_rate)
        / max(1, parallel.pcp)
    )
    eff_prefill = max(1, math.ceil(raw_uncached)) if shape.prefill_tokens > 0 else 0
    rows = decode_rows + eff_prefill
    eff_ranks = (
        parallel.ranks_per_chip
        if parallel.ranks_per_chip is not None
        else chip.cores_per_chip
    )
    p2p = make(
        "pp_p2p",
        "p2p",
        float(rows * geometry.hidden * geometry.hc_mult) * ACTIVATION_BYTES_BF16,
        parallel.pp,
        CollectiveConfig(algo=parallel.ar_algo, ranks_per_chip=eff_ranks),
    )
    return [
        ModuleCost(
            name="pp_p2p",
            hbm_bytes=0.0,
            fp8_flops=0.0,
            bf16_flops=0.0,
            collectives=(p2p,),
            unique_collectives=((p2p, 1),),
        )
    ]


@dataclass(frozen=True, slots=True)
class PpStageOptions:
    """Options controlling stage feasibility, logit peak, and max layer count."""

    max_stage_layers: int | None = None
    logit_peak_b: float = 0.0
    host_table_residency: HostTableResidency = "auto"
    kv_offload_enabled: bool = False


@dataclass(frozen=True, slots=True)
class _PpLayerPrefixes:
    """Prefix sums of per-layer memory footprints."""

    weights: list[float]
    kv: list[float]
    state: list[float]
    embed_b: float
    lm_b: float


@dataclass(frozen=True, slots=True)
class _PpDpProblem:
    """DP state bounds and prefix sums for contiguous pipeline partitioning."""

    pp: int
    n_l: int
    max_stage_layers: int | None
    logit_peak_b: float
    pref: _PpLayerPrefixes


@dataclass(frozen=True, slots=True)
class _DpTables:
    """Dynamic programming cost and backpointer tables."""

    cost: list[list[float]]
    prev: list[list[int]]


def stage_slice_geometry(
    geometry: ModelGeometry,
    start: int,
    end: int,
) -> ModelGeometry:
    """Build a contiguous stage sub-geometry covering layers ``[start, end)``."""
    return slice_geometry_layers(geometry, start, end)


def _layer_has_ple(
    geometry: ModelGeometry,
    ple_set: frozenset[int],
    layer_idx: int,
) -> bool:
    """Return True when layer ``layer_idx`` includes per-layer embedding."""
    if geometry.ple_dim <= 0:
        return False
    if not ple_set:
        return True
    return layer_idx in ple_set


def _layer_extra_key(
    geometry: ModelGeometry,
    layer_idx: int,
) -> tuple[bool, bool, bool, tuple[tuple[bool, bool], ...], bool]:
    """Return per-layer sparse, CED, host-table, and draft membership signature."""
    sparse = geometry.sparse_attn
    if sparse is not None:
        is_kv_src = layer_idx in sparse.kv_source_layer_ids
        is_idx_src = layer_idx in sparse.index_source_layer_ids
        is_ced_post = sparse.ced_post_only or (0 < sparse.ced_boundary <= layer_idx)
    else:
        is_kv_src, is_idx_src, is_ced_post = False, False, False
    tbl_mask = tuple(
        (
            (not spec.layer_ids) or (layer_idx in spec.layer_ids),
            bool(spec.layer_ids) and layer_idx == spec.layer_ids[0],
        )
        for spec in geometry.host_tables
    )
    is_last_draft = geometry.draft is not None and layer_idx == geometry.num_layers - 1
    return (is_kv_src, is_idx_src, is_ced_post, tbl_mask, is_last_draft)


@lru_cache(maxsize=_PP_PREFIX_CACHE_MAXSIZE)
def _pp_layer_prefixes(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    par1: Parallel,
    layout: SeqKvLayout,
    *,
    chip: ChipSpec | None = None,
) -> _PpLayerPrefixes:
    """Compute prefix sums of weights, KV, state, and full-attention counts."""
    num_layers = geometry.num_layers
    tp = max(1, par1.tp)
    emb_single, lm_single = _compute_w_head(geometry, dtypes, tp)
    w_head_single = emb_single + lm_single
    last_stage_lm = emb_single if geometry.tie_word_embeddings else lm_single
    if geometry.mtp_layers > 0 and geometry.draft is None:
        bd_with_mtp = hbm_breakdown_per_core_bytes(
            geometry, dtypes, par1, SeqKvLayout(0, 0), chip=chip
        )
        bd_no_mtp = hbm_breakdown_per_core_bytes(
            replace(geometry, mtp_layers=0), dtypes, par1, SeqKvLayout(0, 0), chip=chip
        )
        w_mtp_total = max(0.0, bd_with_mtp.weights - bd_no_mtp.weights)
    else:
        w_mtp_total = 0.0

    ffn_all = layer_ffn_kinds(geometry)
    ple_set = frozenset(geometry.ple_layer_ids)
    layer_cache: dict[tuple[object, ...], tuple[float, float, float]] = {}
    pref_w = [0.0] * (num_layers + 1)
    pref_kv = [0.0] * (num_layers + 1)
    pref_st = [0.0] * (num_layers + 1)
    for idx in range(num_layers):
        tok = (
            geometry.layer_types[idx]
            if idx < len(geometry.layer_types)
            else "full_attention"
        )
        ffn_kind = ffn_all[idx] if idx < len(ffn_all) else "dense"
        cr_val = (
            geometry.compress_ratios[idx]
            if idx < len(geometry.compress_ratios)
            else None
        )
        idx_val = (
            geometry.indexer_types[idx] if idx < len(geometry.indexer_types) else None
        )
        has_ple = _layer_has_ple(geometry, ple_set, idx)
        lkey: tuple[object, ...] = (
            tok,
            ffn_kind,
            cr_val,
            idx_val,
            has_ple,
            _layer_extra_key(geometry, idx),
        )
        entry = layer_cache.get(lkey)
        if entry is None:
            g_i = stage_slice_geometry(geometry, idx, idx + 1)
            bd_i = hbm_breakdown_per_core_bytes(g_i, dtypes, par1, layout, chip=chip)
            entry = (
                max(0.0, bd_i.weights - w_head_single),
                bd_i.kv,
                bd_i.state,
            )
            layer_cache[lkey] = entry
        pref_w[idx + 1] = pref_w[idx] + entry[0]
        pref_kv[idx + 1] = pref_kv[idx] + entry[1]
        pref_st[idx + 1] = pref_st[idx] + entry[2]
    return _PpLayerPrefixes(
        weights=pref_w,
        kv=pref_kv,
        state=pref_st,
        embed_b=emb_single,
        lm_b=last_stage_lm + w_mtp_total,
    )


def _check_uniform_pp(
    unif: tuple[int, ...],
    prob: _PpDpProblem,
    target_cost: float,
) -> bool:
    """Check if uniform layer partition is legal and cost-competitive."""
    pp = len(unif)
    pref = prob.pref
    u_max = 0.0
    pos = 0
    for s, cnt in enumerate(unif):
        if prob.max_stage_layers is not None and cnt > prob.max_stage_layers:
            return False
        nxt = pos + cnt
        extra_stage_bytes = (pref.embed_b if s == 0 else 0.0) + (
            (pref.lm_b + prob.logit_peak_b) if s == pp - 1 else 0.0
        )
        u_max = max(
            u_max,
            (pref.weights[nxt] - pref.weights[pos])
            + (pref.kv[nxt] - pref.kv[pos])
            + (pref.state[nxt] - pref.state[pos])
            + extra_stage_bytes,
        )
        pos = nxt
    return u_max <= target_cost * (1.0 + _UNIFORM_TIE_TOL)


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
    if prob.pp > prob.n_l:
        return ((), False)
    inf = math.inf
    dp = _DpTables(
        cost=[[inf] * (prob.n_l + 1) for _ in range(prob.pp + 1)],
        prev=[[-1] * (prob.n_l + 1) for _ in range(prob.pp + 1)],
    )
    dp.cost[0][0] = 0.0

    for s in range(1, prob.pp + 1):
        extra = (prob.pref.embed_b if s == 1 else 0.0) + (
            (prob.pref.lm_b + prob.logit_peak_b) if s == prob.pp else 0.0
        )
        _relax_stage_dp(s, extra, prob, dp)

    if dp.prev[prob.pp][prob.n_l] < 0:
        return ((), False)

    rev: list[int] = []
    cur = prob.n_l
    for s in range(prob.pp, 0, -1):
        prv = dp.prev[s][cur]
        rev.append(cur - prv)
        cur = prv
    part = tuple(reversed(rev))

    if prob.n_l % prob.pp == 0:
        unif = (prob.n_l // prob.pp,) * prob.pp
        if _check_uniform_pp(unif, prob, dp.cost[prob.pp][prob.n_l]):
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


class _PpBreakdownsKwargs(TypedDict, total=False):
    """Optional keyword arguments for ``pp_stage_breakdowns``."""

    options: PpStageOptions | None
    chip: ChipSpec | None


class _PpPartitionKwargs(TypedDict, total=False):
    """Optional keyword arguments for ``partition_pp_layers``."""

    options: PpStageOptions | None
    chip: ChipSpec | None


def pp_stage_breakdowns(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    workload: tuple[int, int] = (0, 1),
    **kwargs: Unpack[_PpBreakdownsKwargs],
) -> tuple[tuple[int, ...], HbmBreakdown, float, bool]:
    """Dynamic-programming contiguous PP layer partition and HBM breakdown."""
    context, seqs_per_dp = workload
    opts = kwargs.get("options") or PpStageOptions()
    chip = kwargs.get("chip")
    pp = max(1, parallel.pp)
    num_layers = geometry.num_layers
    layout = SeqKvLayout(
        seqs_per_dp=seqs_per_dp,
        context=context,
        host_table_residency=opts.host_table_residency,
        kv_offload_enabled=opts.kv_offload_enabled,
    )
    if pp <= 1 or num_layers <= 0:
        breakdown = hbm_breakdown_per_core_bytes(
            geometry, dtypes, parallel, layout, chip=chip
        )
        return ((num_layers,), breakdown, breakdown.total, True)
    if pp > num_layers:
        return ((), HbmBreakdown(0.0, 0.0, 0.0, 0.0), 0.0, False)
    par1 = replace(parallel, pp=1)
    pref = _pp_layer_prefixes(geometry, dtypes, par1, layout, chip=chip)
    prob = _PpDpProblem(
        pp=pp,
        n_l=num_layers,
        max_stage_layers=opts.max_stage_layers,
        logit_peak_b=max(0.0, opts.logit_peak_b),
        pref=pref,
    )
    part, feasible = _dp_solve_pp(prob)
    if not feasible or not part:
        return ((), HbmBreakdown(0.0, 0.0, 0.0, 0.0), 0.0, False)
    best_bd, last_tot = _pp_part_breakdown(part, pref, pp)
    return (part, best_bd, last_tot, True)


def partition_pp_layers(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    workload: tuple[int, int] = (0, 1),
    **kwargs: Unpack[_PpPartitionKwargs],
) -> tuple[int, ...]:
    """Dynamic-programming contiguous PP layer partition minimizing max-stage HBM."""
    options = kwargs.get("options")
    chip = kwargs.get("chip")
    eff_opts = options if options is not None else PpStageOptions()
    part, _, _, feasible = pp_stage_breakdowns(
        geometry,
        dtypes,
        parallel,
        workload=workload,
        options=eff_opts,
        chip=chip,
    )
    if not feasible or not part:
        msg = (
            f"Cannot partition {geometry.num_layers} layers into {max(1, parallel.pp)} "
            "non-empty pipeline stages"
        )
        raise ValueError(msg)
    return part


def pp_stage_kv_parts(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    stage_workload: tuple[tuple[int, ...], int],
    *,
    chip: ChipSpec | None = None,
    kv_offload_enabled: bool = False,
) -> tuple[float, float]:
    """Return max-KV-stage (linear_kv_per_token_bytes, fixed_window_kv_bytes)."""
    pp_part, ctx_tokens = stage_workload
    par1 = replace(parallel, pp=1)
    ctx_scale = float(max(1, ctx_tokens))
    best_pair = (0.0, 0.0)
    best_total = -1.0
    pos = 0
    for cnt in pp_part:
        nxt = pos + cnt
        g_stage = stage_slice_geometry(geometry, pos, nxt)
        lin_b, fix_b = kv_bytes_per_core_parts(
            g_stage,
            dtypes,
            par1,
            ctx_tokens,
            chip=chip,
            kv_offload_enabled=kv_offload_enabled,
        )
        stage_tot = lin_b * ctx_scale + fix_b
        if stage_tot > best_total:
            best_total = stage_tot
            best_pair = (lin_b, fix_b)
        pos = nxt
    return best_pair


def _stage_layer_signature(
    geometry: ModelGeometry,
    ffn_all: tuple[str, ...],
    ple_set: frozenset[int],
    pos: int,
    nxt: int,
) -> tuple[object, ...]:
    """Build order-independent per-layer multiset key for stage caching."""
    cnt = nxt - pos
    per_layer: list[tuple[object, ...]] = []
    for idx in range(pos, nxt):
        lt_val = (
            geometry.layer_types[idx]
            if idx < len(geometry.layer_types)
            else "full_attention"
        )
        ffn_val = ffn_all[idx] if idx < len(ffn_all) else "dense"
        cr_raw = (
            geometry.compress_ratios[idx]
            if idx < len(geometry.compress_ratios)
            else None
        )
        idx_raw = (
            geometry.indexer_types[idx] if idx < len(geometry.indexer_types) else None
        )
        has_ple = _layer_has_ple(geometry, ple_set, idx)
        per_layer.append(
            (
                lt_val,
                ffn_val,
                (cr_raw is not None, cr_raw if cr_raw is not None else 0.0),
                (idx_raw is not None, idx_raw if idx_raw is not None else ""),
                has_ple,
                _layer_extra_key(geometry, idx),
            )
        )
    return (cnt, tuple(sorted(Counter(per_layer).items())))


def _merge_stage_floors(
    stage_floors: list[StepFloor],
    *,
    reduce: Literal["max", "sum"] = "max",
) -> StepFloor:
    """Reduce per-stage StepFloor list to bottleneck (max) or serial (sum) metrics."""
    peak_floor = max(
        stage_floors,
        key=lambda floor_result: (
            floor_result.hard_floor_s,
            floor_result.serial_estimate_s,
        ),
    )
    peak = max(
        stage_floors,
        key=lambda floor_result: (
            floor_result.serial_estimate_s,
            floor_result.hard_floor_s,
        ),
    )
    if reduce == "sum":
        return replace(
            peak,
            binding=peak_floor.binding,
            hbm_s=sum(floor_result.hbm_s for floor_result in stage_floors),
            matmul_s=sum(floor_result.matmul_s for floor_result in stage_floors),
            vector_s=sum(floor_result.vector_s for floor_result in stage_floors),
            ici_s=sum(floor_result.ici_s for floor_result in stage_floors),
            chain_s=sum(floor_result.chain_s for floor_result in stage_floors),
            latency_assumed_s=sum(
                floor_result.latency_assumed_s for floor_result in stage_floors
            ),
            hbm_bound_s=sum(floor_result.hbm_bound_s for floor_result in stage_floors),
            matmul_bound_s=sum(
                floor_result.matmul_bound_s for floor_result in stage_floors
            ),
            vector_bound_s=sum(
                floor_result.vector_bound_s for floor_result in stage_floors
            ),
            coll_bound_s=sum(
                floor_result.coll_bound_s for floor_result in stage_floors
            ),
            hard_floor_s=sum(
                floor_result.hard_floor_s for floor_result in stage_floors
            ),
            envelope_floor_s=sum(
                floor_result.envelope_floor_s for floor_result in stage_floors
            ),
            serial_estimate_s=sum(
                floor_result.serial_estimate_s for floor_result in stage_floors
            ),
            pcie_s=sum(floor_result.pcie_s for floor_result in stage_floors),
        )
    return replace(
        peak,
        binding=peak_floor.binding,
        hbm_s=max(floor_result.hbm_s for floor_result in stage_floors),
        matmul_s=max(floor_result.matmul_s for floor_result in stage_floors),
        vector_s=max(floor_result.vector_s for floor_result in stage_floors),
        ici_s=max(floor_result.ici_s for floor_result in stage_floors),
        chain_s=max(floor_result.chain_s for floor_result in stage_floors),
        latency_assumed_s=max(
            floor_result.latency_assumed_s for floor_result in stage_floors
        ),
        hbm_bound_s=max(floor_result.hbm_bound_s for floor_result in stage_floors),
        matmul_bound_s=max(
            floor_result.matmul_bound_s for floor_result in stage_floors
        ),
        vector_bound_s=max(
            floor_result.vector_bound_s for floor_result in stage_floors
        ),
        coll_bound_s=max(floor_result.coll_bound_s for floor_result in stage_floors),
        hard_floor_s=max(floor_result.hard_floor_s for floor_result in stage_floors),
        envelope_floor_s=max(
            floor_result.envelope_floor_s for floor_result in stage_floors
        ),
        serial_estimate_s=max(
            floor_result.serial_estimate_s for floor_result in stage_floors
        ),
        pcie_s=max(floor_result.pcie_s for floor_result in stage_floors),
    )


@lru_cache(maxsize=_PP_STAGE_FLOOR_CACHE_MAXSIZE)
def scale_pp_stage_floor(
    geometry: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    workload: tuple[StepShape, FloorOptions],
    pp_part: tuple[int, ...] | None,
    *,
    reduce: Literal["max", "sum"] = "max",
) -> StepFloor:
    """Evaluate exact per-stage step floors across a contiguous PP layer partition."""
    shape, options = workload
    if pp_part is None or len(pp_part) <= 1 or parallel.pp <= 1:
        return step_floor(geometry, chip, parallel, shape, options)
    if sum(pp_part) != geometry.num_layers or any(cnt <= 0 for cnt in pp_part):
        msg = (
            f"Invalid pp_part {pp_part} for num_layers={geometry.num_layers}, "
            f"pp={parallel.pp}"
        )
        raise ValueError(msg)

    fp8_ratio = chip.fp8_flops / chip.bf16_flops if chip.fp8_flops > 0.0 else 0.0
    handoff_mods = _pp_handoff_mods(geometry, chip, parallel, shape)
    par_stage = replace(parallel, pp=1)
    n_stages = len(pp_part)
    ffn_all = layer_ffn_kinds(geometry)
    ple_set = frozenset(geometry.ple_layer_ids)
    stage_floors: list[StepFloor] = []
    seen_stages: dict[tuple[bool, bool, tuple[object, ...]], StepFloor] = {}
    seen_mods: dict[tuple[object, ...], tuple[ModuleCost, ...]] = {}
    pos = 0
    for stage_idx, cnt in enumerate(pp_part):
        nxt = pos + cnt
        is_first = stage_idx == 0
        is_last = stage_idx == n_stages - 1
        layer_sig = _stage_layer_signature(geometry, ffn_all, ple_set, pos, nxt)
        stage_sig = (is_first, is_last, layer_sig)
        stage_sf = seen_stages.get(stage_sig)
        if stage_sf is None:
            mods = seen_mods.get(layer_sig)
            if mods is None:
                g_stage = stage_slice_geometry(geometry, pos, nxt)
                mods = pass_costs(
                    g_stage,
                    par_stage,
                    shape,
                    options.dtypes,
                    mxu_m_tile=chip.mxu_m_tile,
                    mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
                    mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
                    activation_model=options.activation_model,
                    fp8_to_bf16_ratio=fp8_ratio,
                    chip=chip,
                )
                seen_mods[layer_sig] = mods
            filtered = [
                module
                for module in mods
                if (module.name != "embed" or is_first)
                and (module.name != "lm_head" or is_last)
            ]
            if is_last:
                filtered.extend(
                    prefill_draft_modules(geometry, chip, parallel, shape, options)
                )
            else:
                filtered.extend(handoff_mods)
            stage_sf = combine_modules(
                tuple(filtered),
                chip,
                options.collective_latency_s,
                topology=options.topology,
                parallel=parallel,
            )
            seen_stages[stage_sig] = stage_sf
        stage_floors.append(stage_sf)
        pos = nxt
    return _merge_stage_floors(stage_floors, reduce=reduce)
