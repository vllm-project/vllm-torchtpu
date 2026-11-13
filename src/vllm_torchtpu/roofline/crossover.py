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
"""Analytical prefix-hit crossover sweep, skyline pruning, and EP/PP scaling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from statistics import NormalDist
from typing import TYPE_CHECKING

from .chips import TOPOLOGIES, ChipSpec, SliceTopology
from .constants import EP_PAIR_RANKS, UNIT_GIGA, UNIT_MILLIS_PER_SECOND
from .memory import (
    SeqKvLayout,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
)
from .modules import Dtypes, Parallel, StepShape
from .step import (
    FloorOptions,
    SpecShape,
    StepFloor,
    combine_modules,
    spec_cycle_floor,
    step_floor,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .geometry import ModelGeometry
    from .solver import ParetoPoint, PoolMeshPlan, ServingPlan, TrafficScenario

_STD_NORM = NormalDist()
_INTEGRATION_STEP: float = 1e-3
_INTEGRATION_HALF_SPAN: float = 10.0


@lru_cache(maxsize=256)
def ep_z_unit(ep: int) -> float:
    """Return exact expected maximum of ``ep`` standard normal variables."""
    if ep <= 1:
        return 0.0
    if ep == EP_PAIR_RANKS:
        return 1.0 / math.sqrt(math.pi)
    n_steps = int((2.0 * _INTEGRATION_HALF_SPAN) / _INTEGRATION_STEP)
    n_f = float(ep)
    return sum(
        (-_INTEGRATION_HALF_SPAN + (float(i) + 0.5) * _INTEGRATION_STEP)
        * n_f
        * _STD_NORM.pdf(-_INTEGRATION_HALF_SPAN + (float(i) + 0.5) * _INTEGRATION_STEP)
        * (
            _STD_NORM.cdf(
                -_INTEGRATION_HALF_SPAN + (float(i) + 0.5) * _INTEGRATION_STEP
            )
            ** (ep - 1)
        )
        * _INTEGRATION_STEP
        for i in range(n_steps)
    )


def max_ep_load_ratio(total_assignments: int, ep: int) -> float:
    """Compute expected busiest-rank token-load multiplier under uniform EP routing."""
    if ep <= 1 or total_assignments <= 0:
        return 1.0
    z = ep_z_unit(ep)
    ratio = 1.0 + z * math.sqrt(float(ep - 1) / float(max(1, total_assignments)))
    return min(float(ep), max(1.0, ratio))


@dataclass(frozen=True, slots=True)
class EpLoadSpec:
    """Model, parallelism, and token volume for EP imbalance scaling."""

    geom: ModelGeometry
    par: Parallel
    global_tokens: int


def scale_ep_moe_floor(
    step_floor_val: StepFloor,
    chip: ChipSpec,
    spec: EpLoadSpec,
    *,
    topology: SliceTopology | None = None,
) -> StepFloor:
    """Scale EP MoE module estimate FLOPs by max EP load ratio."""
    if spec.par.ep <= 1 or spec.geom.num_experts <= 0 or spec.global_tokens <= 0:
        return step_floor_val
    ratio = max_ep_load_ratio(spec.global_tokens * spec.geom.top_k, spec.par.ep)
    if ratio <= 1.0:
        return step_floor_val
    scaled = tuple(
        replace(
            m,
            fp8_flops=m.fp8_flops * ratio,
            bf16_flops=m.bf16_flops * ratio,
            fp8_flops_min=m.strict_fp8_flops,
            bf16_flops_min=m.strict_bf16_flops,
        )
        if m.name in ("moe", "draft_moe")
        else m
        for m in step_floor_val.modules
    )
    return combine_modules(scaled, chip, topology=topology, parallel=spec.par)


def scaled_ep_serial_s(
    step_floor_val: StepFloor,
    chip: ChipSpec,
    geom: ModelGeometry,
    parallel: Parallel,
    global_tokens: int,
) -> float:
    """Compute serial execution latency of step floor scaled by EP routing imbalance."""
    if parallel.ep <= 1 or geom.num_experts <= 0 or global_tokens <= 0:
        return step_floor_val.serial_estimate_s
    ratio = max_ep_load_ratio(global_tokens * geom.top_k, parallel.ep)
    if ratio <= 1.0:
        return step_floor_val.serial_estimate_s
    mod_s = 0.0
    for m in step_floor_val.modules:
        scale_factor = ratio if m.name in ("moe", "draft_moe") else 1.0
        hb_s = m.hbm_bytes / chip.hbm_bw
        matmul_s = (m.fp8_flops * scale_factor) / chip.fp8_flops + (
            m.bf16_flops * scale_factor
        ) / chip.bf16_flops
        mod_s += max(hb_s, matmul_s)
    return mod_s + step_floor_val.chain_s


def scale_pp_stage_floor(
    step_floor_val: StepFloor,
    pp_part: tuple[int, ...] | None,
    chip: ChipSpec,
    parallel: Parallel,
    topology: SliceTopology | None = None,
) -> StepFloor:
    """Scale step floor collectives and execution time by uneven PP stage ratio."""
    if not pp_part or len(pp_part) <= 1:
        return step_floor_val
    num_layers = sum(pp_part)
    pp = len(pp_part)
    even_max = max(1, math.ceil(num_layers / pp))
    actual_max = max(pp_part)
    if actual_max == even_max:
        return step_floor_val
    ratio = float(actual_max) / float(even_max)
    scaled = []
    for m in step_floor_val.modules:
        if m.name in ("embed", "lm_head", "p2p", "pp_p2p"):
            scaled.append(m)
            continue
        colls = [c for c in m.collectives if c.name.startswith("pp")]
        non_pp = [c for c in m.collectives if not c.name.startswith("pp")]
        if non_pp:
            target_n = max(1, round(len(non_pp) * ratio))
            colls.extend(non_pp[idx % len(non_pp)] for idx in range(target_n))
        scaled.append(
            replace(
                m,
                hbm_bytes=m.hbm_bytes * ratio,
                fp8_flops=m.fp8_flops * ratio,
                bf16_flops=m.bf16_flops * ratio,
                hbm_bytes_min=(
                    None if m.hbm_bytes_min is None else m.hbm_bytes_min * ratio
                ),
                fp8_flops_min=(
                    None if m.fp8_flops_min is None else m.fp8_flops_min * ratio
                ),
                bf16_flops_min=(
                    None if m.bf16_flops_min is None else m.bf16_flops_min * ratio
                ),
                collectives=tuple(colls),
                unique_collectives=None,
            )
        )
    return combine_modules(tuple(scaled), chip, topology=topology, parallel=parallel)


@dataclass(frozen=True, slots=True)
class MixedCandidateMeta:
    """Precomputed metadata for one mixed candidate in crossover sweep."""

    pt: ParetoPoint
    d_pool: PoolMeshPlan
    d_pid: int
    b_res: int
    cap_ratio: float
    hbm_ratio: float
    pcie_slope: float
    res_cap: float
    conc_f: float
    inv_chips: float
    neg_chips: int


@dataclass(frozen=True, slots=True)
class DisaggTtftItem:
    """Closed-loop disaggregated candidate entry for crossover sweep."""

    p_inst_f: float
    rate_d: float
    inv_chips: float
    tpot_ok: bool
    inter_d: float
    neg_chips: int
    pcie_slope: float
    hbm_ratio: float
    inv_min_c_dcn_ms: float
    dec_dur_s: float
    conc_f: float
    pt: ParetoPoint


@dataclass(frozen=True, slots=True)
class CrossoverPools:
    """Grouped candidate metadata pools for prefix hit rate sweep."""

    mix_meta: tuple[MixedCandidateMeta, ...]
    dis_ttft_groups: dict[
        tuple[int, float], tuple[PoolMeshPlan, float, list[DisaggTtftItem]]
    ]


@dataclass(slots=True)
class CandidateState:
    """Mutable state tracking best throughput candidates across a hit rate sweep."""

    max_slo: float = -1.0
    near_slo: list[tuple[float, float, int, ParetoPoint]] = field(default_factory=list)
    max_all: float = -1.0
    near_all: list[tuple[float, float, int, ParetoPoint]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _GridEvalContext:
    """Context and memoized caches for prefix hit rate sweep evaluation."""

    pre_cache: dict[tuple[int, float], float]
    mix_cache: dict[tuple[int, int, int], float]
    pool_plans: list[tuple[SliceTopology | None, Parallel, tuple[int, ...] | None]]
    compute_pre_sf: Callable[[int, PoolMeshPlan, float], float]
    compute_mix_sf: Callable[[int, int, int], float]
    get_wire_b: Callable[[int], float]
    need_wire: bool
    prefill_chunk: int
    inv_al_ms: float
    dec_steps: float
    think_s: float
    ttft_slo: float | None
    tpot_slo: float | None
    get_dec_sf: Callable[[int, int], tuple[float, float]]


def _pool_sig(pool: PoolMeshPlan) -> tuple[object, ...]:
    """Return canonical layout signature identifying a pool configuration."""
    return (
        pool.topology,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.moe_dispatch,
    )


def _pool_parallel(pool: PoolMeshPlan, topo: SliceTopology | None) -> Parallel:
    """Reconstruct Parallel configuration from pool plan."""
    ranks = topo.cores_per_chip if topo is not None else 2
    return Parallel(
        tp=pool.tp,
        dp=pool.dp,
        ep=pool.ep,
        pp=pool.pp,
        pcp=pool.pcp,
        kv_split=pool.kv_split,
        moe="ep" if pool.ep > 1 else "tp_slice",
        moe_dispatch=pool.moe_dispatch,
        ranks_per_chip=ranks,
        mesh_axes=pool.mesh_axes,
    )


def _calc_sweep_kv_ratios(
    pt: ParetoPoint,
    scen: TrafficScenario,
    b_local: int,
    d_pool: PoolMeshPlan,
    get_b_kv: Callable[[PoolMeshPlan], float],
) -> tuple[float, float, float]:
    """Calculate capacity, HBM, and PCIe transfer ratios matching memory.py."""
    ctx_full = max(1, scen.isl + scen.osl)
    w_tok = max(0.0, scen.kv_working_set_turns) * float(b_local * ctx_full)
    b_kv_phys_core = get_b_kv(d_pool)
    if pt.kv_cache_capacity_tokens is not None and w_tok > 0.0 and b_kv_phys_core > 0.0:
        cap_ratio = min(1.0, max(0.0, float(pt.kv_cache_capacity_tokens) / w_tok))
    else:
        cap_ratio = 1.0

    w_base = d_pool.hbm_weights_gb * UNIT_GIGA
    st_b = d_pool.hbm_state_gb * UNIT_GIGA
    hbm_limit_b = d_pool.hbm_limit_gb * UNIT_GIGA
    cap_hbm = (
        max(0.0, hbm_limit_b - w_base - st_b) / b_kv_phys_core
        if b_kv_phys_core > 0.0
        else 0.0
    )
    hbm_ratio = (
        min(1.0, max(0.0, cap_hbm / w_tok))
        if w_tok > 0.0 and b_kv_phys_core > 0.0
        else 1.0
    )
    return (cap_ratio, hbm_ratio, b_kv_phys_core)


def _eval_mixed_sweep_cand(
    pt: ParetoPoint,
    scen: TrafficScenario,
    get_b_kv: Callable[[PoolMeshPlan], float],
) -> tuple[tuple[object, ...], float, float]:
    """Evaluate mixed candidate constraints and sort keys for sweep pruning."""
    d_pool = pt.decode_pool
    d_idp = max(1, d_pool.instances * max(1, d_pool.dp))
    b_local = max(1, math.ceil(pt.concurrency / d_idp))
    b_res = d_pool.max_num_seqs if d_pool.max_num_seqs is not None else b_local
    cap_ratio, _, _ = _calc_sweep_kv_ratios(pt, scen, b_local, d_pool, get_b_kv)
    has_latency_slo = scen.ttft_slo_ms is not None or scen.tpot_slo_ms is not None
    mkey = (*_pool_sig(d_pool), cap_ratio, pt.concurrency if has_latency_slo else 0)
    over = (
        max(0.0, (float(b_local) / float(max(1, b_res))) - 1.0)
        if scen.ttft_slo_ms is not None
        else 0.0
    )
    return (mkey, pt.token_throughput_per_chip, over)


def _calc_disagg_ttft_dkey(
    pt: ParetoPoint,
    scen: TrafficScenario,
    chip: ChipSpec,
    timing: tuple[int, float, float],
    ratios: tuple[float, float, float, bool],
) -> tuple[tuple[object, ...], float]:
    """Compute equivalence signature and capacity for TTFT-constrained candidate."""
    b_local, dec_dur_s, res_cap_d = timing
    cap_ratio, hbm_ratio, pcie_k, tpot_ok = ratios
    p_pool = pt.prefill_pool
    if p_pool is None:
        msg = "Disaggregated point missing prefill_pool"
        raise RuntimeError(msg)
    d_pool = pt.decode_pool
    min_c = float(max(1, min(p_pool.chips_per_instance, d_pool.chips_per_instance)))
    has_dcn = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
    xfer_k = 1.0 / (min_c * chip.dcn_bw) if has_dcn else 0.0
    pcie_bw = chip.pcie_bw / float(max(1, chip.cores_per_chip))
    has_pcie = pcie_k > 0.0 and pcie_bw > 0.0 and math.isfinite(pcie_bw)
    pcie_slope = pcie_k / pcie_bw if has_pcie else 0.0
    conc_f = float(pt.concurrency)
    c_cap = conc_f / float(pt.total_chips)
    q_wait_d_ms = 0.0
    if scen.ttft_slo_ms is not None:
        d_idp = max(1, d_pool.instances * max(1, d_pool.dp))
        c_dem_d = min(conc_f, float(b_local * d_idp))
        q_wait_d_ms = (
            max(0.0, (c_dem_d / max(1e-9, res_cap_d)) - 1.0)
            * dec_dur_s
            * UNIT_MILLIS_PER_SECOND
        )
    dkey = (
        *_pool_sig(p_pool),
        p_pool.pp_layer_partition,
        cap_ratio,
        round(xfer_k, 8),
        round(pcie_slope, 8),
        hbm_ratio,
        pcie_k,
        tpot_ok,
        q_wait_d_ms,
        dec_dur_s,
        pt.concurrency if scen.ttft_slo_ms is not None else 0,
    )
    return (dkey, c_cap)


def _eval_disagg_sweep_cand(
    pt: ParetoPoint,
    scen: TrafficScenario,
    chip: ChipSpec,
    get_b_kv: Callable[[PoolMeshPlan], float],
) -> tuple[tuple[object, ...], float, float, float, float]:
    """Evaluate disaggregated candidate constraints and sort keys for sweep pruning."""
    p_pool = pt.prefill_pool
    if p_pool is None:
        msg = "Disaggregated point missing prefill_pool"
        raise RuntimeError(msg)
    d_pool = pt.decode_pool
    d_idp = max(1, d_pool.instances * max(1, d_pool.dp))
    b_local = max(1, math.ceil(pt.concurrency / d_idp))
    b_res = d_pool.max_num_seqs if d_pool.max_num_seqs is not None else b_local
    cap_ratio, hbm_ratio, pcie_k = _calc_sweep_kv_ratios(
        pt, scen, b_local, d_pool, get_b_kv
    )
    eff_al = max(1.0, scen.accept_len)
    sf_dec_s = pt.est_tpot_ms * (eff_al * 1e-3)
    dec_dur_s = max(1.0, scen.osl / eff_al) * sf_dec_s
    res_cap_d = float(b_res * d_idp)
    rate_d = min(float(pt.concurrency), res_cap_d) / max(1e-09, dec_dur_s)
    tpot_h = sf_dec_s * (UNIT_MILLIS_PER_SECOND / eff_al)
    inter_h = UNIT_MILLIS_PER_SECOND / max(1e-06, tpot_h)
    tpot_ok = scen.tpot_slo_ms is None or tpot_h <= scen.tpot_slo_ms
    chips_f = float(pt.total_chips)
    seq_tok = float(scen.isl + scen.osl)
    p_cap = float(p_pool.instances) * seq_tok / chips_f
    tput_d_cap = rate_d * seq_tok / chips_f
    dkey, c_cap = _calc_disagg_ttft_dkey(
        pt,
        scen,
        chip,
        (b_local, dec_dur_s, res_cap_d),
        (cap_ratio, hbm_ratio, pcie_k, tpot_ok),
    )
    return (dkey, p_cap, tput_d_cap, c_cap, inter_h)


def _filter_mixed_skyline(
    mixed_groups: dict[tuple[object, ...], list[tuple[float, float, int, ParetoPoint]]],
) -> list[tuple[int, ParetoPoint]]:
    """Filter mixed candidates by minimum latency overhead."""
    kept: list[tuple[int, ParetoPoint]] = []
    for m_items in mixed_groups.values():
        m_items.sort(key=lambda item: (item[0], item[1], item[2]))
        min_over = math.inf
        for _neg_prim, over, idx, pt in m_items:
            if over < min_over:
                min_over = over
                kept.append((idx, pt))
    return kept


def _filter_disagg_3d_skyline(
    d_items: list[tuple[float, float, float, float, int, int, ParetoPoint]],
) -> list[tuple[int, ParetoPoint]]:
    """Filter disaggregated candidates across capacity and interactivity skyline."""
    skyline_3d: list[tuple[int, float, float, float, float]] = []
    kept: list[tuple[int, ParetoPoint]] = []
    for neg_p, neg_d, neg_c, neg_i, chips, idx, pt in d_items:
        p_cap, tput_d, c_cap, inter = -neg_p, -neg_d, -neg_c, -neg_i
        dominated = any(
            pk >= p_cap
            and dk >= tput_d
            and c_k >= c_cap
            and ik >= inter
            and ck <= chips
            for ck, pk, dk, c_k, ik in skyline_3d
        )
        if not dominated:
            kept.append((idx, pt))
            skyline_3d.append((chips, p_cap, tput_d, c_cap, inter))
    return kept


def _filter_sweep_skyline(
    mixed_groups: dict[tuple[object, ...], list[tuple[float, float, int, ParetoPoint]]],
    disagg_groups: dict[
        tuple[object, ...],
        list[tuple[float, float, float, float, int, int, ParetoPoint]],
    ],
) -> tuple[ParetoPoint, ...]:
    """Filter candidate groups using exact multi-dimensional skyline."""
    kept = _filter_mixed_skyline(mixed_groups)
    for d_items in disagg_groups.values():
        d_items.sort(key=lambda item: item[:6])
        kept.extend(_filter_disagg_3d_skyline(d_items))
    kept.sort(key=lambda item: item[0])
    return tuple(pt for _idx, pt in kept)


def prune_sweep_candidates(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scen: TrafficScenario,
    cands: Sequence[ParetoPoint],
) -> tuple[ParetoPoint, ...]:
    """Prune candidates strictly dominated across every prefix hit rate h in [0, 1]."""
    if len(cands) <= 1:
        return tuple(cands)
    kv_b_cache: dict[tuple[int, int, int, int], float] = {}

    def _get_b_kv(pool: PoolMeshPlan) -> float:
        """Compute physical KV bytes per token on one core for a pool."""
        k = (pool.tp, pool.pp, pool.pcp, pool.kv_split)
        val = kv_b_cache.get(k)
        if val is None:
            par_k = Parallel(
                tp=pool.tp, pp=pool.pp, pcp=pool.pcp, kv_split=pool.kv_split
            )
            val = hbm_footprint_per_core_bytes(
                geom, dtypes, par_k, SeqKvLayout(1, 1)
            ) - hbm_footprint_per_core_bytes(geom, dtypes, par_k, SeqKvLayout(1, 0))
            kv_b_cache[k] = val
        return val

    mixed_groups: dict[
        tuple[object, ...], list[tuple[float, float, int, ParetoPoint]]
    ] = {}
    disagg_groups: dict[
        tuple[object, ...],
        list[tuple[float, float, float, float, int, int, ParetoPoint]],
    ] = {}

    for idx, pt in enumerate(cands):
        if pt.mode == "mixed":
            mkey, tput, over = _eval_mixed_sweep_cand(pt, scen, _get_b_kv)
            mixed_groups.setdefault(mkey, []).append((-tput, over, idx, pt))
        else:
            dkey, p_cap, tput_d, c_cap, inter = _eval_disagg_sweep_cand(
                pt, scen, chip, _get_b_kv
            )
            disagg_groups.setdefault(dkey, []).append(
                (-p_cap, -tput_d, -c_cap, -inter, pt.total_chips, idx, pt)
            )

    return _filter_sweep_skyline(mixed_groups, disagg_groups)


def _best_frontier_winner(
    scored: list[tuple[float, float, int, ParetoPoint]],
) -> ParetoPoint:
    """Pick highest-throughput winner from non-dominated frontier candidates."""
    ordered = sorted(scored, key=lambda x: (-x[1], -x[0], -x[2]))
    best_tput = -1.0
    best_pt = ordered[0][3]
    for tput, _inter, _neg_chips, pt in ordered:
        if tput > best_tput * (1.0 + 1e-9):
            best_tput = tput
            best_pt = pt
    return best_pt


def _calc_mix_ttft_and_rate(
    conc_f: float,
    x_capacity: float,
    ttft_service_s: float,
    t_dec_s: float,
    think_s: float,
) -> tuple[float, float]:
    """Calculate TTFT in ms and request rate for mixed serving."""
    tot_round_s = ttft_service_s + t_dec_s + think_s
    rate_s = min(x_capacity, conc_f / max(1e-09, tot_round_s))
    ttft_s = conc_f / max(1e-09, rate_s) - t_dec_s - think_s
    return (ttft_s * UNIT_MILLIS_PER_SECOND, rate_s)


def _record_grid_candidate(
    state: CandidateState,
    cand: tuple[float, float, int, ParetoPoint],
    *,
    slo_ok: bool,
) -> None:
    """Record candidate into SLO or fallback frontier bucket."""
    tol_band = 1.0 + 1e-7
    tput = cand[0]
    if slo_ok:
        if tput * tol_band >= state.max_slo:
            state.max_slo = max(state.max_slo, tput)
            state.near_slo.append(cand)
    elif not state.near_slo and tput * tol_band >= state.max_all:
        state.max_all = max(state.max_all, tput)
        state.near_all.append(cand)


def _eval_one_mixed_meta(
    meta: MixedCandidateMeta,
    h: float,
    scen: TrafficScenario,
    eval_context: _GridEvalContext,
    defaults: tuple[int, int],
) -> tuple[tuple[float, float, int, ParetoPoint], bool]:
    """Evaluate one mixed candidate metadata item at prefix hit rate h."""
    n_chk_1, chunk_tok_1 = defaults
    if meta.cap_ratio == 1.0 or h == 0.0:
        h_eff, n_chk, chunk_tok_c = h, n_chk_1, chunk_tok_1
    else:
        h_eff = h * meta.cap_ratio
        eff_isl_h = max(1, math.ceil((1.0 - h_eff) * scen.isl))
        n_chk = max(1, math.ceil(eff_isl_h / eval_context.prefill_chunk))
        chunk_tok_c = min(eval_context.prefill_chunk, eff_isl_h)
    pcie_s = (
        float(scen.isl) * max(0.0, h_eff - h * meta.hbm_ratio) * meta.pcie_slope
        if meta.pcie_slope > 0.0
        else 0.0
    )
    sf_pre_s = eval_context.pre_cache.get((meta.d_pid, h_eff))
    if sf_pre_s is None:
        sf_pre_s = eval_context.compute_pre_sf(meta.d_pid, meta.d_pool, h_eff)
    ttft_service_s = sf_pre_s + pcie_s

    d_dp = max(1, meta.d_pool.dp)
    max_seqs = float(meta.b_res * d_dp)
    b_dec = max(1.0, min(max_seqs, meta.conc_f))
    est_step_s = 0.0
    t_dec_s = 0.0
    x_capacity = 0.0
    for _ in range(3):
        b_rep = max(1, min(meta.b_res, math.ceil(b_dec / float(d_dp))))
        sf_dec_s_step, draft_s_step = eval_context.get_dec_sf(meta.d_pid, b_rep)
        dec_dur_s_step = eval_context.dec_steps * sf_dec_s_step
        sf_mix_s = eval_context.mix_cache.get((meta.d_pid, b_rep, chunk_tok_c))
        if sf_mix_s is None:
            sf_mix_s = eval_context.compute_mix_sf(meta.d_pid, b_rep, chunk_tok_c)
        mix_serial_s = sf_mix_s + draft_s_step
        bn = float(b_rep * n_chk)
        tot_work_s = (
            dec_dur_s_step
            + float(meta.b_res * n_chk) * mix_serial_s
            + float(meta.b_res) * pcie_s
        )
        p_mix = min(1.0, bn / max(1.0, eval_context.dec_steps + bn))
        est_step_s = (1.0 - p_mix) * sf_dec_s_step + p_mix * mix_serial_s
        t_dec_s = eval_context.dec_steps * est_step_s
        x_capacity = meta.res_cap / max(1e-09, tot_work_s)
        _ttft_tmp, rate_s = _calc_mix_ttft_and_rate(
            meta.conc_f, x_capacity, ttft_service_s, t_dec_s, eval_context.think_s
        )
        b_dec = max(1.0, min(max_seqs, rate_s * t_dec_s))

    ttft_h, rate_s = _calc_mix_ttft_and_rate(
        meta.conc_f, x_capacity, ttft_service_s, t_dec_s, eval_context.think_s
    )
    tpot_h = est_step_s * eval_context.inv_al_ms
    inter_h = UNIT_MILLIS_PER_SECOND / max(1e-06, tpot_h)
    tput = rate_s * float(scen.isl + scen.osl) * meta.inv_chips
    slo_ok = (eval_context.ttft_slo is None or ttft_h <= eval_context.ttft_slo) and (
        eval_context.tpot_slo is None or tpot_h <= eval_context.tpot_slo
    )
    return ((tput, inter_h, meta.neg_chips, meta.pt), slo_ok)


def eval_grid_mix_candidates(
    h: float,
    scen: TrafficScenario,
    mix_meta: Sequence[MixedCandidateMeta],
    eval_context: _GridEvalContext,
    state: CandidateState,
) -> None:
    """Evaluate mixed candidate throughput and latency under prefix hit rate h."""
    eff_isl_1 = max(1, math.ceil((1.0 - h) * scen.isl))
    n_chk_1 = max(1, math.ceil(eff_isl_1 / eval_context.prefill_chunk))
    chunk_tok_1 = min(eval_context.prefill_chunk, eff_isl_1)
    for meta in mix_meta:
        cand, slo_ok = _eval_one_mixed_meta(
            meta, h, scen, eval_context, (n_chk_1, chunk_tok_1)
        )
        _record_grid_candidate(state, cand, slo_ok=slo_ok)


def _eval_grid_disagg_ttft(
    h: float,
    scen: TrafficScenario,
    dis_ttft_groups: dict[
        tuple[int, float], tuple[PoolMeshPlan, float, list[DisaggTtftItem]]
    ],
    eval_context: _GridEvalContext,
    state: CandidateState,
) -> None:
    """Evaluate closed-loop disaggregated candidates under prefix hit rate h."""
    eff_isl_1 = max(1, math.ceil((1.0 - h) * scen.isl))
    n_chk_1 = max(1, math.ceil(eff_isl_1 / eval_context.prefill_chunk))
    wire_b_1 = eval_context.get_wire_b(eff_isl_1) if eval_context.need_wire else 0.0

    for (p_pid, cap_ratio), (p_pool, bub_num, grp_t) in dis_ttft_groups.items():
        if cap_ratio == 1.0 or h == 0.0:
            h_eff, n_chk, wire_b_h = h, n_chk_1, wire_b_1
        else:
            h_eff = h * cap_ratio
            eff_isl_h = max(1, math.ceil((1.0 - h_eff) * scen.isl))
            n_chk = max(1, math.ceil(eff_isl_h / eval_context.prefill_chunk))
            wire_b_h = (
                eval_context.get_wire_b(eff_isl_h) if eval_context.need_wire else 0.0
            )
        sf_pre_s = eval_context.pre_cache.get((p_pid, h_eff))
        if sf_pre_s is None:
            sf_pre_s = eval_context.compute_pre_sf(p_pid, p_pool, h_eff)
        denom_pre = max(1e-09, sf_pre_s)
        bub = 1.0 + bub_num / float(n_chk) if bub_num > 0.0 else 1.0
        pre_bub_s = sf_pre_s * bub
        for item in grp_t:
            rate_p = item.p_inst_f / denom_pre
            coupled_cap = min(rate_p, item.rate_d)
            pcie_s = (
                float(scen.isl) * max(0.0, h_eff - h * item.hbm_ratio) * item.pcie_slope
                if item.pcie_slope > 0.0
                else 0.0
            )
            xfer_ms = wire_b_h * item.inv_min_c_dcn_ms
            ttft_service_s = pre_bub_s + pcie_s + xfer_ms * 1e-3
            coupled = min(
                coupled_cap,
                item.conc_f
                / max(1e-09, ttft_service_s + item.dec_dur_s + eval_context.think_s),
            )
            ttft_h = (
                item.conc_f / max(1e-09, coupled)
                - item.dec_dur_s
                - eval_context.think_s
            ) * UNIT_MILLIS_PER_SECOND
            tput = coupled * float(scen.isl + scen.osl) * item.inv_chips
            slo_ok = item.tpot_ok and (
                eval_context.ttft_slo is None or ttft_h <= eval_context.ttft_slo
            )
            _record_grid_candidate(
                state, (tput, item.inter_d, item.neg_chips, item.pt), slo_ok=slo_ok
            )


def _eval_grid_h(
    h: float,
    scen: TrafficScenario,
    pools: CrossoverPools,
    eval_context: _GridEvalContext,
) -> (
    tuple[
        float,
        tuple[tuple[object, ...], tuple[object, ...] | None, int, float, ParetoPoint],
    ]
    | None
):
    """Find winning candidate configuration for hit rate h across all pools."""
    state = CandidateState()
    eval_grid_mix_candidates(h, scen, pools.mix_meta, eval_context, state)
    _eval_grid_disagg_ttft(h, scen, pools.dis_ttft_groups, eval_context, state)
    tol_band = 1.0 + 1e-7
    candidates = (
        [x for x in state.near_slo if x[0] * tol_band >= state.max_slo]
        if state.near_slo
        else [x for x in state.near_all if x[0] * tol_band >= state.max_all]
    )
    if not candidates:
        return None
    winner = _best_frontier_winner(candidates)
    return (
        h,
        (
            _pool_sig(winner.decode_pool),
            _pool_sig(winner.prefill_pool) if winner.prefill_pool else None,
            winner.total_chips,
            winner.token_throughput_per_chip,
            winner,
        ),
    )


def _layout_sig_tuple(
    pt: ParetoPoint,
) -> tuple[tuple[object, ...], tuple[object, ...] | None, int]:
    """Return structural layout signature matching solver diversity rules."""
    return (
        _pool_sig(pt.decode_pool),
        _pool_sig(pt.prefill_pool) if pt.prefill_pool else None,
        pt.total_chips,
    )


def _derive_winner_range_and_crossover(
    scen: TrafficScenario,
    plan: ServingPlan,
    winners: list[
        tuple[
            float,
            tuple[
                tuple[object, ...], tuple[object, ...] | None, int, float, ParetoPoint
            ],
        ]
    ],
) -> tuple[tuple[float, float] | None, float | None]:
    """Derive validity range and crossover hit rate for recommended plan."""
    if plan.recommended is None:
        return (None, None)
    h0 = round(float(scen.prefix_hit_rate), 4)
    idx0 = min(range(len(winners)), key=lambda i: abs(winners[i][0] - h0))
    sig0 = _layout_sig_tuple(plan.recommended)
    sigs = [(h, (d_sig, p_sig, chips)) for h, (d_sig, p_sig, chips, _, _) in winners]
    sigs[idx0] = (sigs[idx0][0], sig0)
    lo = idx0
    while lo > 0 and sigs[lo - 1][1] == sig0:
        lo -= 1
    hi = idx0
    while hi + 1 < len(sigs) and sigs[hi + 1][1] == sig0:
        hi += 1
    win_range = (sigs[lo][0], sigs[hi][0])

    cross: float | None = None
    best_dist = math.inf
    for i in range(len(sigs) - 1):
        h_a, sig_a = sigs[i]
        h_b, sig_b = sigs[i + 1]
        if sig_a != sig_b:
            mid = round(0.5 * (h_a + h_b), 4)
            dist = abs(mid - h0)
            if dist < best_dist:
                best_dist = dist
                cross = mid
    return (win_range, cross)


class _CrossoverPlanner:
    """Precompute candidate pool metadata and step floor caches for crossover sweep."""

    def __init__(
        self,
        geom: ModelGeometry,
        dtypes: Dtypes,
        chip: ChipSpec,
        scen: TrafficScenario,
    ) -> None:
        """Initialize cache containers and compute static parameters."""
        self.geom = geom
        self.dtypes = dtypes
        self.chip = chip
        self.scen = scen
        self.ctx_mid: int = scen.isl + scen.osl // 2
        self.eff_al = max(1.0, scen.accept_len)
        self.spec_k = (
            scen.spec_tokens
            if scen.spec_tokens is not None
            else (
                max(geom.mtp_layers, math.ceil(self.eff_al) - 1)
                if self.eff_al > 1.0
                else 0
            )
        )
        self.dec_steps = max(1.0, scen.osl / self.eff_al)
        has_dcn = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
        self.think_s = scen.think_time_s
        self.need_wire = has_dcn

        self.pool_ids: dict[tuple[object, ...], int] = {}
        self.pool_plans: list[
            tuple[SliceTopology | None, Parallel, tuple[int, ...] | None]
        ] = []
        self.pre_cache: dict[tuple[int, float], float] = {}
        self.pre_step_cache: dict[tuple[int, int, int, int], float] = {}
        self.dec_cache: dict[tuple[int, int], tuple[float, float]] = {}
        self.mix_cache: dict[tuple[int, int, int], float] = {}
        self.kv_b_cache: dict[tuple[int, int, int, int], float] = {}
        self._init_wire()

    def _init_wire(self) -> None:
        """Initialize unpadded KV wire transfer bytes when DCN is enabled."""
        geom = self.geom
        self.sw_w = geom.sliding_window
        if not self.need_wire:
            self.wire_state_b = 0.0
            self.wire_kv_step1 = 0.0
            self.wire_kv_at_sw = 0.0
            self.wire_kv_post_sw = 0.0
            return
        par_w = Parallel(tp=1, pp=1)
        bd_w0 = hbm_breakdown_per_core_bytes(
            geom,
            self.dtypes,
            par_w,
            SeqKvLayout(1, 0, physical_kv_layout=False),
        )
        bd_w1 = hbm_breakdown_per_core_bytes(
            geom,
            self.dtypes,
            par_w,
            SeqKvLayout(1, 1, physical_kv_layout=False),
        )
        self.wire_state_b, self.wire_kv_step1 = bd_w0.state, bd_w1.kv
        if self.sw_w > 0:
            bd_w_sw = hbm_breakdown_per_core_bytes(
                geom,
                self.dtypes,
                par_w,
                SeqKvLayout(1, self.sw_w, physical_kv_layout=False),
            )
            bd_w_sw1 = hbm_breakdown_per_core_bytes(
                geom,
                self.dtypes,
                par_w,
                SeqKvLayout(1, self.sw_w + 1, physical_kv_layout=False),
            )
            self.wire_kv_at_sw = bd_w_sw.kv
            self.wire_kv_post_sw = bd_w_sw1.kv - bd_w_sw.kv
        else:
            self.wire_kv_at_sw, self.wire_kv_post_sw = 0.0, 0.0

    def get_wire_b(self, uncached_tok: int) -> float:
        """Compute KV cache wire transfer bytes for uncached prefix tokens."""
        tok_k = max(0, uncached_tok)
        if self.sw_w > 0 and tok_k > self.sw_w:
            kv_w = self.wire_kv_at_sw + float(tok_k - self.sw_w) * self.wire_kv_post_sw
        else:
            kv_w = float(tok_k) * self.wire_kv_step1
        return self.wire_state_b + kv_w

    def get_b_kv(self, pool: PoolMeshPlan) -> float:
        """Calculate single-token physical KV cache bytes per core for a pool."""
        k = (pool.tp, pool.pp, pool.pcp, pool.kv_split)
        val = self.kv_b_cache.get(k)
        if val is None:
            par_k = Parallel(
                tp=pool.tp, pp=pool.pp, pcp=pool.pcp, kv_split=pool.kv_split
            )
            val = hbm_footprint_per_core_bytes(
                self.geom, self.dtypes, par_k, SeqKvLayout(1, 1)
            ) - hbm_footprint_per_core_bytes(
                self.geom, self.dtypes, par_k, SeqKvLayout(1, 0)
            )
            self.kv_b_cache[k] = val
        return val

    def get_pool_id(self, pool: PoolMeshPlan) -> int:
        """Allocate or look up cached index for a PoolMeshPlan."""
        pk: tuple[object, ...] = (
            *_pool_sig(pool),
            pool.mesh_axes,
            pool.pp_layer_partition,
        )
        pid = self.pool_ids.get(pk)
        if pid is None:
            pid = len(self.pool_plans)
            self.pool_ids[pk] = pid
            topo = TOPOLOGIES.get(pool.topology)
            parallel = _pool_parallel(pool, topo)
            max_p = max(1, math.ceil(self.geom.num_layers / max(1, pool.pp)))
            even_pp = (
                pool.pp <= 1
                or not pool.pp_layer_partition
                or max(pool.pp_layer_partition) == max_p
            )
            self.pool_plans.append(
                (topo, parallel, None if even_pp else pool.pp_layer_partition)
            )
        return pid

    def compute_pre_sf(self, pid: int, pool: PoolMeshPlan, h_val: float) -> float:
        """Compute unbubbled prefill step floor latency under prefix caching."""
        h_clamped = max(0.0, min(1.0, h_val))
        eff_tok = (
            max(1, math.ceil((1.0 - h_clamped) * self.scen.isl))
            if self.scen.isl > 0
            else 0
        )
        eff_pre = (
            max(
                1,
                math.ceil(self.scen.isl * max(0.0, 1.0 - h_clamped) / max(1, pool.pcp)),
            )
            if self.scen.isl > 0
            else 0
        )
        cached_pre = round(self.scen.isl * h_clamped) if self.scen.isl > 0 else 0
        skey = (pid, eff_tok, eff_pre, cached_pre)
        sf_s = self.pre_step_cache.get(skey)
        if sf_s is None:
            topo, parallel, pp_part = self.pool_plans[pid]
            shape = StepShape(
                batch=0,
                rows_per_seq=0,
                context=0,
                state_writes=0,
                prefill_tokens=self.scen.isl,
                prefix_hit_rate=h_clamped,
            )
            sf_obj = step_floor(
                self.geom,
                self.chip,
                parallel,
                shape,
                FloorOptions(dt=self.dtypes, topology=topo),
            )
            if parallel.pp > 1:
                sf_obj = scale_pp_stage_floor(
                    sf_obj, pp_part, self.chip, parallel, topo
                )
            sf_s = sf_obj.serial_estimate_s
            self.pre_step_cache[skey] = sf_s
        return sf_s

    def _eval_spec_dec_sf(
        self,
        pid: int,
        b_res: int,
    ) -> tuple[float, float]:
        """Compute speculative cycle and draft delta seconds for decode batch."""
        topo, parallel, pp_part = self.pool_plans[pid]
        opts = FloorOptions(dt=self.dtypes, topology=topo)
        sf_dec = spec_cycle_floor(
            self.geom,
            self.chip,
            parallel,
            SpecShape(batch=b_res, context=self.ctx_mid, k=self.spec_k),
            opts,
        )
        sf_ver_only = step_floor(
            self.geom,
            self.chip,
            parallel,
            StepShape(
                batch=b_res,
                rows_per_seq=self.spec_k + 1,
                context=self.ctx_mid,
                state_writes=1,
            ),
            opts,
        )
        if parallel.pp > 1:
            sf_dec = scale_pp_stage_floor(sf_dec, pp_part, self.chip, parallel, topo)
            sf_ver_only = scale_pp_stage_floor(
                sf_ver_only, pp_part, self.chip, parallel, topo
            )
        if parallel.ep > 1:
            tot_tok = b_res * (self.spec_k + 1)
            ep_spec = EpLoadSpec(geom=self.geom, par=parallel, global_tokens=tot_tok)
            sf_dec = scale_ep_moe_floor(sf_dec, self.chip, ep_spec, topology=topo)
            sf_ver_only = scale_ep_moe_floor(
                sf_ver_only, self.chip, ep_spec, topology=topo
            )
        draft_s = max(0.0, sf_dec.serial_estimate_s - sf_ver_only.serial_estimate_s)
        return (sf_dec.serial_estimate_s, draft_s)

    def get_dec_sf(self, pid: int, b_res: int) -> tuple[float, float]:
        """Retrieve or compute decode step floor latency."""
        key = (pid, b_res)
        val = self.dec_cache.get(key)
        if val is not None:
            return val
        if self.spec_k > 0:
            val = self._eval_spec_dec_sf(pid, b_res)
            self.dec_cache[key] = val
            return val
        topo, parallel, pp_part = self.pool_plans[pid]
        step_floor_val = step_floor(
            self.geom,
            self.chip,
            parallel,
            StepShape(batch=b_res, rows_per_seq=1, context=self.ctx_mid),
            FloorOptions(dt=self.dtypes, topology=topo),
        )
        if parallel.pp > 1:
            step_floor_val = scale_pp_stage_floor(
                step_floor_val, pp_part, self.chip, parallel, topo
            )
        val = (
            scaled_ep_serial_s(step_floor_val, self.chip, self.geom, parallel, b_res),
            0.0,
        )
        self.dec_cache[key] = val
        return val

    def compute_mix_sf(self, pid: int, b_res: int, chunk_tok: int) -> float:
        """Compute mixed prefill-decode step floor latency."""
        key = (pid, b_res, chunk_tok)
        val = self.mix_cache.get(key)
        if val is None:
            topo, parallel, pp_part = self.pool_plans[pid]
            shape = StepShape(
                batch=b_res,
                rows_per_seq=self.spec_k + 1,
                context=self.ctx_mid,
                state_writes=1,
                prefill_tokens=chunk_tok,
                prefix_hit_rate=0.0,
            )
            step_floor_val = step_floor(
                self.geom,
                self.chip,
                parallel,
                shape,
                FloorOptions(dt=self.dtypes, topology=topo),
            )
            if parallel.pp > 1:
                step_floor_val = scale_pp_stage_floor(
                    step_floor_val, pp_part, self.chip, parallel, topo
                )
            val = scaled_ep_serial_s(
                step_floor_val,
                self.chip,
                self.geom,
                parallel,
                b_res * (self.spec_k + 1) + chunk_tok,
            )
            self.mix_cache[key] = val
        return val

    def add_mixed_cand(
        self, pt: ParetoPoint, d_pid: int, mix_meta: list[MixedCandidateMeta]
    ) -> None:
        """Append precomputed MixedCandidateMeta for one mixed candidate."""
        d_pool = pt.decode_pool
        d_dp = max(1, d_pool.dp)
        b_local = max(1, math.ceil(pt.concurrency / max(1, d_pool.instances * d_dp)))
        b_res = d_pool.max_num_seqs if d_pool.max_num_seqs is not None else b_local
        cap_ratio, hbm_ratio, pcie_k = _calc_sweep_kv_ratios(
            pt, self.scen, b_local, d_pool, self.get_b_kv
        )
        pcie_core_bw = self.chip.pcie_bw / float(max(1, self.chip.cores_per_chip))
        has_pcie = pcie_k > 0.0 and pcie_core_bw > 0.0 and math.isfinite(pcie_core_bw)
        mix_meta.append(
            MixedCandidateMeta(
                pt=pt,
                d_pool=d_pool,
                d_pid=d_pid,
                b_res=b_res,
                cap_ratio=cap_ratio,
                hbm_ratio=hbm_ratio,
                pcie_slope=(pcie_k / pcie_core_bw) if has_pcie else 0.0,
                res_cap=float(b_res * d_dp),
                conc_f=float(pt.concurrency),
                inv_chips=1.0 / float(pt.total_chips),
                neg_chips=-pt.total_chips,
            )
        )

    def add_disagg_cand(
        self,
        pt: ParetoPoint,
        d_pid: int,
        dis_ttft_groups: dict[
            tuple[int, float], tuple[PoolMeshPlan, float, list[DisaggTtftItem]]
        ],
    ) -> None:
        """Append precomputed closed-loop item for one disaggregated candidate."""
        p_pool = pt.prefill_pool
        if p_pool is None:
            msg = "Disaggregated point missing prefill_pool"
            raise RuntimeError(msg)
        d_pool = pt.decode_pool
        d_idp = max(1, d_pool.instances * max(1, d_pool.dp))
        b_local = max(1, math.ceil(pt.concurrency / d_idp))
        b_res = d_pool.max_num_seqs if d_pool.max_num_seqs is not None else b_local
        p_pid = self.get_pool_id(p_pool)
        cap_ratio, hbm_ratio, pcie_k = _calc_sweep_kv_ratios(
            pt, self.scen, b_local, d_pool, self.get_b_kv
        )
        sf_dec_s, _draft_s = self.get_dec_sf(d_pid, b_res)
        inv_al_ms = UNIT_MILLIS_PER_SECOND / self.eff_al
        dec_dur_s = self.dec_steps * sf_dec_s
        conc_f = float(pt.concurrency)
        rate_d = min(conc_f, float(b_res * d_idp)) / max(1e-09, dec_dur_s)
        tpot_d = sf_dec_s * inv_al_ms
        inter_d = UNIT_MILLIS_PER_SECOND / max(1e-06, tpot_d)
        tpot_ok = self.scen.tpot_slo_ms is None or tpot_d <= self.scen.tpot_slo_ms
        gkey = (p_pid, cap_ratio)
        pcie_core_bw = self.chip.pcie_bw / float(max(1, self.chip.cores_per_chip))
        has_pcie = pcie_k > 0.0 and pcie_core_bw > 0.0 and math.isfinite(pcie_core_bw)
        has_dcn = self.chip.dcn_bw > 0.0 and math.isfinite(self.chip.dcn_bw)
        min_c = float(max(1, min(p_pool.chips_per_instance, d_pool.chips_per_instance)))
        inv_min_c_dcn_ms = (
            UNIT_MILLIS_PER_SECOND / (min_c * self.chip.dcn_bw) if has_dcn else 0.0
        )
        dis_ttft_groups.setdefault(gkey, (p_pool, float(p_pool.pp - 1), []))[2].append(
            DisaggTtftItem(
                p_inst_f=float(p_pool.instances),
                rate_d=rate_d,
                inv_chips=1.0 / float(pt.total_chips),
                tpot_ok=tpot_ok,
                inter_d=inter_d,
                neg_chips=-pt.total_chips,
                pcie_slope=(pcie_k / pcie_core_bw) if has_pcie else 0.0,
                hbm_ratio=hbm_ratio,
                inv_min_c_dcn_ms=inv_min_c_dcn_ms,
                dec_dur_s=dec_dur_s,
                conc_f=conc_f,
                pt=pt,
            )
        )


def setup_crossover_sweep(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scen: TrafficScenario,
    cands: Sequence[ParetoPoint],
) -> tuple[CrossoverPools, _GridEvalContext]:
    """Precompute candidate pool metadata and step floor caches for crossover sweep."""
    planner = _CrossoverPlanner(geom, dtypes, chip, scen)
    mix_meta: list[MixedCandidateMeta] = []
    dis_ttft_groups: dict[
        tuple[int, float], tuple[PoolMeshPlan, float, list[DisaggTtftItem]]
    ] = {}

    for pt in cands:
        d_pid = planner.get_pool_id(pt.decode_pool)
        if pt.mode == "mixed":
            planner.add_mixed_cand(pt, d_pid, mix_meta)
        else:
            planner.add_disagg_cand(pt, d_pid, dis_ttft_groups)

    pools = CrossoverPools(
        mix_meta=tuple(mix_meta),
        dis_ttft_groups=dis_ttft_groups,
    )
    eval_context = _GridEvalContext(
        pre_cache=planner.pre_cache,
        mix_cache=planner.mix_cache,
        pool_plans=planner.pool_plans,
        compute_pre_sf=planner.compute_pre_sf,
        compute_mix_sf=planner.compute_mix_sf,
        get_wire_b=planner.get_wire_b,
        need_wire=planner.need_wire,
        prefill_chunk=scen.prefill_chunk,
        inv_al_ms=UNIT_MILLIS_PER_SECOND / planner.eff_al,
        dec_steps=planner.dec_steps,
        think_s=planner.think_s,
        ttft_slo=scen.ttft_slo_ms,
        tpot_slo=scen.tpot_slo_ms,
        get_dec_sf=planner.get_dec_sf,
    )
    return (pools, eval_context)


def sweep_prefix_hit_crossovers(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    plan: ServingPlan,
    *,
    hit_rates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95),
) -> tuple[tuple[float, float] | None, float | None]:
    """Sweep candidate HBM-fit pools across prefix hit rates."""
    if plan.recommended is None:
        return (None, None)
    raw_cands = (
        plan.sweep_candidates
        if plan.sweep_candidates
        else (plan.candidates if plan.candidates else plan.frontier)
    )
    if not raw_cands:
        return (None, None)
    scen = plan.scenario
    cands = (
        raw_cands
        if plan.sweep_candidates
        else prune_sweep_candidates(geom, dtypes, chip, scen, raw_cands)
    )
    grid = tuple(
        sorted(
            {
                *(round(float(h), 4) for h in hit_rates),
                round(scen.prefix_hit_rate, 4),
            }
        )
    )
    pools, eval_context = setup_crossover_sweep(geom, dtypes, chip, scen, cands)
    winners = [
        entry
        for h in grid
        if (entry := _eval_grid_h(h, scen, pools, eval_context)) is not None
    ]
    if not winners:
        return (None, None)
    return _derive_winner_range_and_crossover(scen, plan, winners)
