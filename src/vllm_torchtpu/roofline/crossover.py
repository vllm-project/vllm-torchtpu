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
"""Heuristic serial-estimate prefix-hit crossover sweep, pruning, and EP scaling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from statistics import NormalDist
from typing import TYPE_CHECKING, Final, TypedDict, Unpack

from .chips import ChipSpec, SliceTopology, topologies_for_chip
from .constants import (
    CAUSAL_MIDPOINT_HALF,
    DEFAULT_CORES_PER_CHIP,
    EP_PAIR_RANKS,
    UNIT_GIBI,
    UNIT_GIGA,
    UNIT_MILLI,
    UNIT_MILLIS_PER_SECOND,
)
from .memory import (
    SeqKvLayout,
    hbm_breakdown_per_core_bytes,
    kv_bytes_per_core_parts,
)
from .modules import Dtypes, ExpertActivationModel, Parallel, StepShape
from .partition import pp_stage_kv_parts, scale_pp_stage_floor
from .schema import (
    _point_struct_key as _schema_point_struct_key,
)
from .schema import (
    resolve_spec_k,
    shared_prefix_tokens,
)
from .step import (
    FloorOptions,
    SpecShape,
    StepFloor,
    combine_modules,
    module_serial_s,
    spec_cycle_floor,
    step_floor,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .geometry import ModelGeometry
    from .schema import (
        ParetoPoint,
        PoolMeshPlan,
        ServingPlan,
        SortBy,
        TrafficScenario,
    )

_STD_NORM = NormalDist()
_INTEGRATION_STEP: Final[float] = 1e-3
_INTEGRATION_HALF_SPAN: Final[float] = 10.0
_RATE_EPS: Final[float] = 1e-9
_TPOT_EPS: Final[float] = 1e-6
_TOL_BAND: Final[float] = 1.0 + 1e-7
_WINNER_TIE_TOL: Final[float] = 1.0 + 1e-9
_KEY_ROUND_DIGITS: Final[int] = 6
_HIT_ROUND_DIGITS: Final[int] = 4
_FIXED_POINT_ITERS: Final[int] = 3
_DEFAULT_HIT_RATES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)


@lru_cache(maxsize=256)
def ep_z_unit(ep_ranks: int) -> float:
    """Return exact expected maximum of ``ep_ranks`` standard normal variables."""
    if ep_ranks <= 1:
        return 0.0
    if ep_ranks == EP_PAIR_RANKS:
        return 1.0 / math.sqrt(math.pi)
    num_steps = int((2.0 * _INTEGRATION_HALF_SPAN) / _INTEGRATION_STEP)
    ep_ranks_f = float(ep_ranks)
    return sum(
        (-_INTEGRATION_HALF_SPAN + (float(step_idx) + 0.5) * _INTEGRATION_STEP)
        * ep_ranks_f
        * _STD_NORM.pdf(
            -_INTEGRATION_HALF_SPAN + (float(step_idx) + 0.5) * _INTEGRATION_STEP
        )
        * (
            _STD_NORM.cdf(
                -_INTEGRATION_HALF_SPAN + (float(step_idx) + 0.5) * _INTEGRATION_STEP
            )
            ** (ep_ranks - 1)
        )
        * _INTEGRATION_STEP
        for step_idx in range(num_steps)
    )


def max_ep_load_ratio(total_assignments: int, ep_ranks: int) -> float:
    """Compute expected busiest-rank token-load multiplier under uniform EP routing."""
    if ep_ranks <= 1 or total_assignments <= 0:
        return 1.0
    std_normal_max = ep_z_unit(ep_ranks)
    ratio = 1.0 + std_normal_max * math.sqrt(
        float(ep_ranks - 1) / float(max(1, total_assignments))
    )
    return min(float(ep_ranks), max(1.0, ratio))


def scaled_ep_serial_s(
    step_floor_val: StepFloor,
    chip: ChipSpec,
    geometry: ModelGeometry,
    parallel: Parallel,
    global_tokens: int,
) -> float:
    """Compute serial execution latency of step floor scaled by EP routing imbalance."""
    if parallel.ep <= 1 or geometry.num_experts <= 0 or global_tokens <= 0:
        return step_floor_val.serial_estimate_s
    ratio = max_ep_load_ratio(global_tokens * geometry.top_k, parallel.ep)
    if ratio <= 1.0:
        return step_floor_val.serial_estimate_s
    mod_s = 0.0
    for module in step_floor_val.modules:
        if module.name in ("moe", "draft_moe"):
            scaled_module = replace(module, vector_flops=module.vector_flops * ratio)
            mod_s += module_serial_s(scaled_module, chip, matmul_scale=ratio)
        else:
            mod_s += module_serial_s(module, chip)
    return mod_s + step_floor_val.chain_s


@dataclass(frozen=True, slots=True)
class _SweepKvRequest:
    """Explicit pool, working-set batch, and context inputs for KV ratio checks."""

    pool: PoolMeshPlan
    working_set_batch: int
    ctx_len: int
    b_res_fixed: int
    kv_cap_tokens: int | None


@dataclass(frozen=True, slots=True)
class MixedCandidateMeta:
    """Precomputed metadata for one mixed candidate in crossover sweep."""

    point: ParetoPoint
    d_pool: PoolMeshPlan
    decode_pool_id: int
    resident_batch: int
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

    prefill_dp_instances_f: float
    rate_d: float
    inv_chips: float
    tpot_ok: bool
    inter_d: float
    neg_chips: int
    pcie_slope: float
    hbm_ratio: float
    inv_min_c_dcn_ms: float
    dcn_scale: float
    dec_dur_s: float
    conc_f: float
    point: ParetoPoint
    queue_wait_d_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class CrossoverPools:
    """Grouped candidate metadata pools for prefix hit rate sweep."""

    mix_meta: tuple[MixedCandidateMeta, ...]
    dis_ttft_groups: dict[
        tuple[int, float], tuple[PoolMeshPlan, float, list[DisaggTtftItem]]
    ]


@dataclass(slots=True)
class CandidateState:
    """Mutable state tracking best candidates across a hit rate sweep."""

    max_slo: float = -1.0
    near_slo: list[tuple[float, float, int, ParetoPoint]] = field(default_factory=list)
    max_all: float = -1.0
    near_all: list[tuple[float, float, int, ParetoPoint]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _GridEvalContext:
    """Context and memoized caches for prefix hit rate sweep evaluation."""

    mix_cache: dict[tuple[int, int, int, float, int], float]
    pure_chunk_cache: dict[tuple[int, int, float], float]
    pool_plans: list[tuple[SliceTopology | None, Parallel, tuple[int, ...] | None]]
    compute_pre_sf: Callable[[int, PoolMeshPlan, float], float]
    compute_mix_sf: Callable[[int, int, int, float, int, int], float]
    compute_pure_chunk_sf: Callable[[int, int, float], float]
    get_wire_b: Callable[[int], float]
    need_wire: bool
    prefill_chunk: int
    inv_al_ms: float
    dec_steps: float
    think_s: float | None
    ttft_slo: float | None
    tpot_slo: float | None
    get_dec_sf: Callable[[int, int], tuple[float, float]]
    spec_k: int = 0
    sort_by: SortBy = "throughput"


def _pool_sig(pool: PoolMeshPlan) -> tuple[object, ...]:
    """Return canonical layout signature identifying a pool configuration."""
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.moe_dispatch,
    )


def _resolve_topo_map(
    chip: ChipSpec,
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None = None,
) -> dict[str, SliceTopology]:
    """Resolve slice topology lookup map, preferring explicit topologies."""
    if topologies is None:
        return topologies_for_chip(chip)
    if isinstance(topologies, Mapping):
        return dict(topologies)
    return {topology.name: topology for topology in topologies}


_MESH_AXES_LEN: Final[int] = 4


def _pool_parallel(
    pool: PoolMeshPlan,
    topology: SliceTopology | None,
    *,
    default_ranks: int = DEFAULT_CORES_PER_CHIP,
) -> Parallel:
    """Reconstruct Parallel configuration from pool plan."""
    ranks = topology.cores_per_chip if topology is not None else default_ranks
    raw_axes = pool.mesh_axes
    mesh_axes: tuple[int, int, int, int] | None
    if not raw_axes:
        mesh_axes = None
    elif len(raw_axes) == _MESH_AXES_LEN:
        mesh_axes = (raw_axes[0], raw_axes[1], raw_axes[2], raw_axes[3])
    else:
        msg = f"Expected empty or {_MESH_AXES_LEN}-tuple mesh_axes, got {raw_axes!r}"
        raise ValueError(msg)
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
        mesh_axes=mesh_axes,
        residual=pool.residual,
    )


def _valid_uneven_pp(
    pool: PoolMeshPlan,
    num_layers: int,
) -> tuple[int, ...] | None:
    """Return pool.pp_layer_partition only when it is a valid uneven stage split."""
    part = pool.pp_layer_partition
    if (
        pool.pp <= 1
        or not part
        or len(part) != pool.pp
        or sum(part) != num_layers
        or any(cnt <= 0 for cnt in part)
        or len(set(part)) <= 1
    ):
        return None
    return part


def _resolve_pp_partition(
    pool: PoolMeshPlan,
    num_layers: int,
) -> tuple[int, ...] | None:
    """Return validated uneven or uniform stage partition for ``pool.pp > 1``."""
    uneven = _valid_uneven_pp(pool, num_layers)
    if uneven is not None:
        return uneven
    if pool.pp > 1 and num_layers % pool.pp == 0:
        return (num_layers // pool.pp,) * pool.pp
    return None


def _kv_parts_for_pool(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    pool: PoolMeshPlan,
    ctx_len: int,
) -> tuple[float, float]:
    """Return (linear_kv_per_token_bytes, fixed_window_kv_bytes) on one core."""
    par_k = Parallel(
        tp=pool.tp,
        pp=pool.pp,
        pcp=pool.pcp,
        kv_split=pool.kv_split,
        ranks_per_chip=chip.cores_per_chip,
        residual=pool.residual,
    )
    pp_part = _resolve_pp_partition(pool, geometry.num_layers)
    if pp_part is not None:
        return pp_stage_kv_parts(geometry, dtypes, par_k, (pp_part, ctx_len), chip=chip)
    return kv_bytes_per_core_parts(geometry, dtypes, par_k, ctx_len, chip=chip)


def _calc_sweep_kv_ratios(
    req: _SweepKvRequest,
    scenario: TrafficScenario,
    get_b_kv_parts: Callable[[PoolMeshPlan, int], tuple[float, float]],
) -> tuple[float, float, float]:
    """Calculate capacity, HBM, and linear KV bytes/token matching memory.py."""
    pool = req.pool
    ctx_len = max(1, req.ctx_len)
    working_set_tokens = max(0.0, scenario.kv_working_set_turns) * float(
        req.working_set_batch * ctx_len
    )
    b_kv_phys_core, fixed_kv_b = get_b_kv_parts(pool, ctx_len)
    w_base = pool.hbm_weights_gb * UNIT_GIGA
    state_bytes = pool.hbm_state_gb * UNIT_GIGA
    hbm_limit_b = pool.hbm_limit_gb * UNIT_GIGA
    offload_core_b = (
        float(pool.kv_offload_gb)
        * UNIT_GIBI
        / float(max(1, pool.tp * pool.pp * pool.dp * pool.pcp))
    )

    if b_kv_phys_core > 0.0:
        fixed_res_b = float(max(0, req.b_res_fixed)) * fixed_kv_b
        avail_hbm_b = max(0.0, hbm_limit_b - w_base - state_bytes - fixed_res_b)
        cap_hbm = avail_hbm_b / b_kv_phys_core
        if req.kv_cap_tokens is not None and working_set_tokens > 0.0:
            cap_ratio = min(
                1.0, max(0.0, float(req.kv_cap_tokens) / max(1.0, working_set_tokens))
            )
        elif working_set_tokens > 0.0:
            cap_total = (avail_hbm_b + offload_core_b) / b_kv_phys_core
            cap_ratio = min(1.0, max(0.0, cap_total / max(1.0, working_set_tokens)))
        else:
            cap_ratio = 1.0
        hbm_ratio = (
            min(1.0, max(0.0, cap_hbm / max(1.0, working_set_tokens)))
            if working_set_tokens > 0.0
            else 1.0
        )
        return (cap_ratio, hbm_ratio, b_kv_phys_core)

    if fixed_kv_b > 0.0:
        working_set_seqs = max(0.0, scenario.kv_working_set_turns) * float(
            max(0, req.working_set_batch)
        )
        avail_hbm_b = max(0.0, hbm_limit_b - w_base - state_bytes)
        cap_seqs_tot = (avail_hbm_b + offload_core_b) / fixed_kv_b
        cap_seqs_hbm = avail_hbm_b / fixed_kv_b
        if working_set_seqs > 0.0:
            cap_ratio = min(1.0, max(0.0, cap_seqs_tot / max(1.0, working_set_seqs)))
            hbm_ratio = min(1.0, max(0.0, cap_seqs_hbm / max(1.0, working_set_seqs)))
        else:
            cap_ratio = 1.0
            hbm_ratio = 1.0
        return (cap_ratio, hbm_ratio, 0.0)

    return (1.0, 1.0, 0.0)


def _eval_mixed_sweep_cand(
    point: ParetoPoint,
    scenario: TrafficScenario,
    get_b_kv_parts: Callable[[PoolMeshPlan, int], tuple[float, float]],
) -> tuple[tuple[object, ...], float, float]:
    """Evaluate mixed candidate constraints and sort keys for sweep pruning."""
    d_pool = point.decode_pool
    decode_dp_instances = max(1, d_pool.instances * max(1, d_pool.dp))
    local_batch = max(1, math.ceil(point.concurrency / decode_dp_instances))
    resident_batch = (
        d_pool.resident_batch_cap
        if d_pool.resident_batch_cap is not None
        else local_batch
    )
    req = _SweepKvRequest(
        pool=d_pool,
        working_set_batch=local_batch,
        ctx_len=max(1, scenario.isl + scenario.osl),
        b_res_fixed=resident_batch,
        kv_cap_tokens=point.kv_cache_capacity_tokens,
    )
    cap_ratio, hbm_ratio, _ = _calc_sweep_kv_ratios(req, scenario, get_b_kv_parts)
    mixed_group_key = (
        *_pool_sig(d_pool),
        d_pool.pp_layer_partition,
        resident_batch,
        round(cap_ratio, _KEY_ROUND_DIGITS),
        round(hbm_ratio, _KEY_ROUND_DIGITS),
        d_pool.kv_offload_gb,
        d_pool.hbm_weights_gb,
        point.concurrency,
    )
    over = (
        max(0.0, (float(local_batch) / float(max(1, resident_batch))) - 1.0)
        if scenario.ttft_slo_ms is not None
        else 0.0
    )
    return (mixed_group_key, point.token_throughput_per_chip, over)


def _calc_disagg_ttft_dkey(
    point: ParetoPoint,
    scenario: TrafficScenario,
    chip: ChipSpec,
    timing: tuple[int, float, float],
    ratios: tuple[float, float, float, bool],
) -> tuple[tuple[object, ...], float]:
    """Compute equivalence signature and capacity for TTFT-constrained candidate."""
    local_batch, dec_dur_s, res_cap_d = timing
    cap_ratio, hbm_ratio, pcie_k, tpot_ok = ratios
    p_pool = point.prefill_pool
    if p_pool is None:
        msg = "Disaggregated point missing prefill_pool"
        raise RuntimeError(msg)
    d_pool = point.decode_pool
    min_c = max(1, min(p_pool.chips_per_instance, d_pool.chips_per_instance))
    min_pool = min(
        p_pool.instances * p_pool.chips_per_instance,
        d_pool.instances * d_pool.chips_per_instance,
    )
    has_dcn = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
    pcie_bw = chip.pcie_bw / float(max(1, chip.cores_per_chip))
    has_pcie = pcie_k > 0.0 and pcie_bw > 0.0 and math.isfinite(pcie_bw)
    conc_f = float(point.concurrency)
    c_cap = conc_f / float(point.total_chips)
    q_wait_d_ms = 0.0
    if scenario.ttft_slo_ms is not None:
        decode_dp_instances = max(1, d_pool.instances * max(1, d_pool.dp))
        c_dem_d = min(conc_f, float(local_batch * decode_dp_instances))
        q_wait_d_ms = (
            max(0.0, (c_dem_d / max(_RATE_EPS, res_cap_d)) - 1.0)
            * dec_dur_s
            * UNIT_MILLIS_PER_SECOND
        )
    disagg_group_key = (
        *_pool_sig(p_pool),
        p_pool.pp_layer_partition,
        cap_ratio,
        (min_c, min_pool, has_dcn),
        (pcie_k, pcie_bw if has_pcie else 0.0),
        hbm_ratio,
        pcie_k,
        tpot_ok,
        q_wait_d_ms,
        dec_dur_s,
        point.concurrency if scenario.ttft_slo_ms is not None else 0,
    )
    return (disagg_group_key, c_cap)


def _eval_disagg_sweep_cand(
    point: ParetoPoint,
    scenario: TrafficScenario,
    chip: ChipSpec,
    get_b_kv_parts: Callable[[PoolMeshPlan, int], tuple[float, float]],
) -> tuple[tuple[object, ...], float, float, float, float]:
    """Evaluate disaggregated candidate constraints and sort keys for sweep pruning."""
    p_pool = point.prefill_pool
    if p_pool is None:
        msg = "Disaggregated point missing prefill_pool"
        raise RuntimeError(msg)
    d_pool = point.decode_pool
    decode_dp_instances = max(1, d_pool.instances * max(1, d_pool.dp))
    local_batch = max(1, math.ceil(point.concurrency / decode_dp_instances))
    resident_batch = (
        d_pool.resident_batch_cap
        if d_pool.resident_batch_cap is not None
        else local_batch
    )
    prefill_dp_instances = max(1, p_pool.instances * max(1, p_pool.dp))
    working_set_prefill = max(1, math.ceil(point.concurrency / prefill_dp_instances))
    req = _SweepKvRequest(
        pool=p_pool,
        working_set_batch=working_set_prefill,
        ctx_len=max(1, scenario.isl),
        b_res_fixed=1,
        kv_cap_tokens=None,
    )
    cap_ratio, hbm_ratio, pcie_k = _calc_sweep_kv_ratios(req, scenario, get_b_kv_parts)
    effective_accept_len = max(1.0, scenario.accept_len)
    sf_dec_s = point.est_tpot_ms * (effective_accept_len * UNIT_MILLI)
    dec_dur_s = (float(max(0, scenario.osl - 1)) / effective_accept_len) * sf_dec_s
    res_cap_d = float(resident_batch * decode_dp_instances)
    rate_d = min(float(point.concurrency), res_cap_d) / max(_RATE_EPS, dec_dur_s)
    tpot_h = sf_dec_s * (UNIT_MILLIS_PER_SECOND / effective_accept_len)
    inter_h = UNIT_MILLIS_PER_SECOND / max(_TPOT_EPS, tpot_h)
    tpot_ok = scenario.tpot_slo_ms is None or tpot_h <= scenario.tpot_slo_ms
    chips_f = float(point.total_chips)
    seq_tok = float(scenario.isl + scenario.osl)
    p_cap = float(prefill_dp_instances) * seq_tok / chips_f
    tput_d_cap = rate_d * seq_tok / chips_f
    disagg_group_key, c_cap = _calc_disagg_ttft_dkey(
        point,
        scenario,
        chip,
        (local_batch, dec_dur_s, res_cap_d),
        (cap_ratio, hbm_ratio, pcie_k, tpot_ok),
    )
    return (disagg_group_key, p_cap, tput_d_cap, c_cap, inter_h)


def _pareto_prune_mixed(
    mixed_groups: dict[tuple[object, ...], list[tuple[float, float, int, ParetoPoint]]],
) -> list[tuple[int, ParetoPoint]]:
    """Filter mixed candidates by minimum latency overhead."""
    kept: list[tuple[int, ParetoPoint]] = []
    for m_items in mixed_groups.values():
        m_items.sort(key=lambda item: (item[0], item[1], item[2]))
        min_over = math.inf
        for _neg_prim, over, idx, point in m_items:
            if over < min_over:
                min_over = over
                kept.append((idx, point))
    return kept


def _pareto_prune_disagg_3d(
    d_items: list[tuple[float, float, float, float, int, int, ParetoPoint]],
) -> list[tuple[int, ParetoPoint]]:
    """Filter disaggregated candidates across capacity and interactivity frontier."""
    pareto_3d: list[tuple[int, float, float, float, float]] = []
    kept: list[tuple[int, ParetoPoint]] = []
    for neg_p, neg_d, neg_c, neg_i, chips, idx, point in d_items:
        p_cap, tput_d, c_cap, inter = -neg_p, -neg_d, -neg_c, -neg_i
        dominated = any(
            prefill_cap_k >= p_cap
            and decode_cap_k >= tput_d
            and conc_cap_k >= c_cap
            and inter_k >= inter
            and chips_k <= chips
            for chips_k, prefill_cap_k, decode_cap_k, conc_cap_k, inter_k in pareto_3d
        )
        if not dominated:
            kept.append((idx, point))
            pareto_3d.append((chips, p_cap, tput_d, c_cap, inter))
    return kept


def _pareto_prune_sweep(
    mixed_groups: dict[tuple[object, ...], list[tuple[float, float, int, ParetoPoint]]],
    disagg_groups: dict[
        tuple[object, ...],
        list[tuple[float, float, float, float, int, int, ParetoPoint]],
    ],
) -> tuple[ParetoPoint, ...]:
    """Filter candidate groups using exact multi-dimensional Pareto dominance."""
    kept = _pareto_prune_mixed(mixed_groups)
    for d_items in disagg_groups.values():
        d_items.sort(key=lambda item: item[:6])
        kept.extend(_pareto_prune_disagg_3d(d_items))
    kept.sort(key=lambda item: item[0])
    return tuple(point for _idx, point in kept)


def prune_sweep_candidates(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    cands: Sequence[ParetoPoint],
) -> tuple[ParetoPoint, ...]:
    """Prune candidates strictly dominated across every prefix hit rate in [0, 1]."""
    if len(cands) <= 1:
        return tuple(cands)
    kv_b_cache: dict[
        tuple[int, int, int, int, str, tuple[int, ...] | None, int],
        tuple[float, float],
    ] = {}

    def _get_b_kv_parts(pool: PoolMeshPlan, ctx_len: int) -> tuple[float, float]:
        """Compute physical linear and fixed-window KV bytes on one core for a pool."""
        kv_cache_key = (
            pool.tp,
            pool.pp,
            pool.pcp,
            pool.kv_split,
            pool.residual,
            pool.pp_layer_partition,
            ctx_len,
        )
        cached_value = kv_b_cache.get(kv_cache_key)
        if cached_value is None:
            cached_value = _kv_parts_for_pool(geometry, dtypes, chip, pool, ctx_len)
            kv_b_cache[kv_cache_key] = cached_value
        return cached_value

    mixed_groups: dict[
        tuple[object, ...], list[tuple[float, float, int, ParetoPoint]]
    ] = {}
    disagg_groups: dict[
        tuple[object, ...],
        list[tuple[float, float, float, float, int, int, ParetoPoint]],
    ] = {}

    for idx, point in enumerate(cands):
        if point.mode == "mixed":
            mixed_group_key, tput, over = _eval_mixed_sweep_cand(
                point, scenario, _get_b_kv_parts
            )
            mixed_groups.setdefault(mixed_group_key, []).append(
                (-tput, over, idx, point)
            )
        else:
            disagg_group_key, p_cap, tput_d, c_cap, inter = _eval_disagg_sweep_cand(
                point, scenario, chip, _get_b_kv_parts
            )
            disagg_groups.setdefault(disagg_group_key, []).append(
                (-p_cap, -tput_d, -c_cap, -inter, point.total_chips, idx, point)
            )

    return _pareto_prune_sweep(mixed_groups, disagg_groups)


def _best_frontier_winner(
    scored: list[tuple[float, float, int, ParetoPoint]],
    sort_by: SortBy = "throughput",
) -> ParetoPoint:
    """Pick deterministic frontier winner with total-order structural tie-break."""
    if sort_by == "interactivity":
        ordered = sorted(
            scored,
            key=lambda candidate: (
                -candidate[0],
                -candidate[1],
                -candidate[2],
                _point_struct_key(candidate[3]),
            ),
        )
        best_inter = -1.0
        best_point = ordered[0][3]
        for _tput, inter, _neg_chips, point in ordered:
            if inter > best_inter * _WINNER_TIE_TOL:
                best_inter = inter
                best_point = point
        return best_point
    ordered = sorted(
        scored,
        key=lambda candidate: (
            -candidate[1],
            -candidate[0],
            -candidate[2],
            _point_struct_key(candidate[3]),
        ),
    )
    best_tput = -1.0
    best_point = ordered[0][3]
    for tput, _inter, _neg_chips, point in ordered:
        if tput > best_tput * _WINNER_TIE_TOL:
            best_tput = tput
            best_point = point
    return best_point


def _calc_mix_ttft_and_rate(
    conc_f: float,
    x_capacity: float,
    ttft_service_s: float,
    t_dec_s: float,
    think_s: float | None,
    *,
    res_cap: float,
) -> tuple[float, float]:
    """Calculate TTFT in ms and request rate for mixed serving."""
    if think_s is None:
        served_frac = min(1.0, conc_f / max(_RATE_EPS, res_cap))
        rate_s = served_frac * x_capacity
        service_time_s = ttft_service_s + t_dec_s
        queue_wait_s = (
            max(0.0, (conc_f / max(_RATE_EPS, res_cap)) - 1.0) * service_time_s
        )
        ttft_s = ttft_service_s + queue_wait_s
        return (ttft_s * UNIT_MILLIS_PER_SECOND, rate_s)
    tot_round_s = ttft_service_s + t_dec_s + think_s
    rate_s = min(x_capacity, conc_f / max(_RATE_EPS, tot_round_s))
    ttft_s = max(ttft_service_s, conc_f / max(_RATE_EPS, rate_s) - t_dec_s - think_s)
    return (ttft_s * UNIT_MILLIS_PER_SECOND, rate_s)


def _primary_metric(
    cand: tuple[float, float, int, ParetoPoint],
    sort_by: SortBy,
) -> float:
    """Return active primary ranking metric (throughput or interactivity)."""
    return cand[1] if sort_by == "interactivity" else cand[0]


def _record_grid_candidate(
    state: CandidateState,
    cand: tuple[float, float, int, ParetoPoint],
    *,
    slo_ok: bool,
    sort_by: SortBy = "throughput",
) -> None:
    """Record candidate into SLO or fallback frontier bucket."""
    primary = _primary_metric(cand, sort_by)
    if slo_ok:
        if primary * _TOL_BAND >= state.max_slo:
            state.max_slo = max(state.max_slo, primary)
            state.near_slo.append(cand)
    elif not state.near_slo and primary * _TOL_BAND >= state.max_all:
        state.max_all = max(state.max_all, primary)
        state.near_all.append(cand)


def _mixed_chunk_context_stats(
    hit_rate: float,
    cap_ratio: float,
    isl: int,
    prefill_chunk: int,
    defaults: tuple[int, int],
) -> tuple[float, int, int, int, float]:
    """Return (eff_hit, num_chunks, chunk_tok_c, mix_pre_total, mix_hit_rate)."""
    num_chunks_default, chunk_tok_default = defaults
    if cap_ratio == 1.0 or hit_rate == 0.0:
        eff_hit, num_chunks, chunk_tok_c = (
            hit_rate,
            num_chunks_default,
            chunk_tok_default,
        )
    else:
        eff_hit = hit_rate * cap_ratio
        eff_isl_tmp = max(1, math.ceil((1.0 - eff_hit) * isl))
        num_chunks = max(1, math.ceil(eff_isl_tmp / prefill_chunk))
        chunk_tok_c = min(prefill_chunk, eff_isl_tmp)
    eff_isl_h = max(1, math.ceil((1.0 - eff_hit) * isl))
    cached_prefix = max(0, isl - eff_isl_h)
    avg_prior = float(cached_prefix) + CAUSAL_MIDPOINT_HALF * float(
        max(0, eff_isl_h - chunk_tok_c)
    )
    mix_pre_total = max(1, round(avg_prior + float(chunk_tok_c)))
    mix_hit_rate = min(1.0, max(0.0, avg_prior / float(mix_pre_total)))
    return (eff_hit, num_chunks, chunk_tok_c, mix_pre_total, mix_hit_rate)


def _mixed_replica_req_time_s(
    meta: MixedCandidateMeta,
    eval_context: _GridEvalContext,
    step_s: tuple[float, float, float],
    chunk_stats: tuple[int, int, float],
) -> float:
    """Compute per-replica request service duration for one mixed iteration."""
    sf_dec_s_step, mix_serial_s, pcie_s = step_s
    num_chunks, mix_pre_total, mix_hit_rate = chunk_stats
    dec_per_res = eval_context.dec_steps / float(max(1, meta.resident_batch))
    num_chunks_f = float(num_chunks)
    if dec_per_res >= num_chunks_f:
        return (
            (dec_per_res - num_chunks_f) * sf_dec_s_step
            + num_chunks_f * mix_serial_s
            + pcie_s
        )
    pure_chunk_s = eval_context.pure_chunk_cache.get(
        (meta.decode_pool_id, mix_pre_total, mix_hit_rate)
    )
    if pure_chunk_s is None:
        pure_chunk_s = eval_context.compute_pure_chunk_sf(
            meta.decode_pool_id, mix_pre_total, mix_hit_rate
        )
    pp_factor = 1.0 + float(meta.d_pool.pp - 1) / float(max(1, num_chunks))
    return (
        dec_per_res * mix_serial_s
        + (num_chunks_f - dec_per_res) * pure_chunk_s * pp_factor
        + pcie_s
    )


def _mixed_pure_decode_upper_bound(
    meta: MixedCandidateMeta,
    num_chunks: int,
    scenario: TrafficScenario,
    eval_context: _GridEvalContext,
) -> float | None:
    """Return sound pure-decode throughput upper bound for EP=1, PP=1, spec_k=0."""
    if (
        meta.d_pool.kv_offload_gb > 0.0
        or eval_context.dec_steps <= 0.0
        or meta.d_pool.ep != 1
        or meta.d_pool.pp != 1
        or eval_context.spec_k != 0
    ):
        return None
    dec_per_res = eval_context.dec_steps / float(max(1, meta.resident_batch))
    if dec_per_res < float(num_chunks):
        return None
    decode_dp_instances = max(1, meta.d_pool.instances * max(1, meta.d_pool.dp))
    batch_upper_bound = 1
    sf_dec_ub, _ = eval_context.get_dec_sf(meta.decode_pool_id, batch_upper_bound)
    t_dec_ub = eval_context.dec_steps * sf_dec_ub
    x_cap_ub = float(decode_dp_instances * meta.resident_batch) / max(
        _RATE_EPS, t_dec_ub
    )
    think_floor = 0.0 if eval_context.think_s is None else eval_context.think_s
    rate_ub = min(x_cap_ub, meta.conc_f / max(_RATE_EPS, t_dec_ub + think_floor))
    return rate_ub * float(scenario.isl + scenario.osl) * meta.inv_chips


def _eval_one_mixed_meta(
    meta: MixedCandidateMeta,
    hit_rate: float,
    scenario: TrafficScenario,
    eval_context: _GridEvalContext,
    defaults: tuple[int, int],
) -> tuple[tuple[float, float, int, ParetoPoint], bool]:
    """Evaluate one mixed candidate estimate throughput at prefix hit rate."""
    eff_hit, num_chunks, chunk_tok_c, mix_pre_total, mix_hit_rate = (
        _mixed_chunk_context_stats(
            hit_rate, meta.cap_ratio, scenario.isl, eval_context.prefill_chunk, defaults
        )
    )
    pcie_s = (
        float(scenario.isl)
        * max(0.0, eff_hit - hit_rate * meta.hbm_ratio)
        * meta.pcie_slope
        if meta.pcie_slope > 0.0
        else 0.0
    )
    sf_pre_s = eval_context.compute_pre_sf(meta.decode_pool_id, meta.d_pool, eff_hit)
    pp_factor = 1.0 + float(meta.d_pool.pp - 1) / float(max(1, num_chunks))
    ttft_service_s = sf_pre_s * pp_factor + pcie_s
    shared_tok = round(eff_hit * scenario.isl)

    decode_dp_instances = max(1, meta.d_pool.instances * max(1, meta.d_pool.dp))
    max_seqs = float(meta.resident_batch * decode_dp_instances)
    b_dec = max(1.0, min(max_seqs, meta.conc_f))
    est_step_s = 0.0
    t_dec_s = 0.0
    x_capacity = 0.0
    prev_b_rep = -1
    for _ in range(_FIXED_POINT_ITERS):
        b_rep = max(
            1, min(meta.resident_batch, math.ceil(b_dec / float(decode_dp_instances)))
        )
        if b_rep == prev_b_rep:
            break
        prev_b_rep = b_rep
        sf_dec_s_step, draft_s_step = eval_context.get_dec_sf(
            meta.decode_pool_id, b_rep
        )
        sf_mix_s = eval_context.mix_cache.get(
            (meta.decode_pool_id, b_rep, mix_pre_total, mix_hit_rate, shared_tok)
        )
        if sf_mix_s is None:
            sf_mix_s = eval_context.compute_mix_sf(
                meta.decode_pool_id,
                b_rep,
                mix_pre_total,
                mix_hit_rate,
                chunk_tok_c,
                shared_tok,
            )
        mix_serial_s = sf_mix_s + draft_s_step
        p_mix = (
            min(1.0, float(b_rep * num_chunks) / eval_context.dec_steps)
            if eval_context.dec_steps > 0.0
            else 0.0
        )
        est_step_s = (1.0 - p_mix) * sf_dec_s_step + p_mix * mix_serial_s
        t_dec_s = eval_context.dec_steps * est_step_s
        replica_req_time_s = _mixed_replica_req_time_s(
            meta,
            eval_context,
            (sf_dec_s_step, mix_serial_s, pcie_s),
            (num_chunks, mix_pre_total, mix_hit_rate),
        )
        x_capacity = float(decode_dp_instances) / max(_RATE_EPS, replica_req_time_s)
        _ttft_tmp, rate_s = _calc_mix_ttft_and_rate(
            meta.conc_f,
            x_capacity,
            ttft_service_s,
            t_dec_s,
            eval_context.think_s,
            res_cap=meta.res_cap,
        )
        b_dec = max(1.0, min(max_seqs, rate_s * t_dec_s))

    ttft_h, rate_s = _calc_mix_ttft_and_rate(
        meta.conc_f,
        x_capacity,
        ttft_service_s,
        t_dec_s,
        eval_context.think_s,
        res_cap=meta.res_cap,
    )
    tpot_h = est_step_s * eval_context.inv_al_ms
    inter_h = UNIT_MILLIS_PER_SECOND / max(_TPOT_EPS, tpot_h)
    tput = rate_s * float(scenario.isl + scenario.osl) * meta.inv_chips
    slo_ok = (eval_context.ttft_slo is None or ttft_h <= eval_context.ttft_slo) and (
        eval_context.tpot_slo is None or tpot_h <= eval_context.tpot_slo
    )
    return ((tput, inter_h, meta.neg_chips, meta.point), slo_ok)


def _eval_grid_disagg_ttft(
    hit_rate: float,
    scenario: TrafficScenario,
    dis_ttft_groups: dict[
        tuple[int, float], tuple[PoolMeshPlan, float, list[DisaggTtftItem]]
    ],
    eval_context: _GridEvalContext,
    state: CandidateState,
) -> None:
    """Evaluate disaggregated candidates under prefix hit rate."""
    raw_uncached_1 = max(0, math.ceil((1.0 - hit_rate) * scenario.isl))
    eff_isl_1 = max(1, raw_uncached_1)
    num_chunks_1 = max(1, math.ceil(eff_isl_1 / eval_context.prefill_chunk))
    wire_b_1 = (
        eval_context.get_wire_b(raw_uncached_1)
        if (eval_context.need_wire and raw_uncached_1 > 0)
        else 0.0
    )

    for (prefill_pool_id, cap_ratio), (
        p_pool,
        pp_bubble_stages,
        group_items,
    ) in dis_ttft_groups.items():
        if cap_ratio == 1.0 or hit_rate == 0.0:
            eff_hit, num_chunks, wire_b_h = hit_rate, num_chunks_1, wire_b_1
        else:
            eff_hit = hit_rate * cap_ratio
            raw_uncached_h = max(0, math.ceil((1.0 - eff_hit) * scenario.isl))
            eff_isl_h = max(1, raw_uncached_h)
            num_chunks = max(1, math.ceil(eff_isl_h / eval_context.prefill_chunk))
            wire_b_h = (
                eval_context.get_wire_b(raw_uncached_h)
                if (eval_context.need_wire and raw_uncached_h > 0)
                else 0.0
            )
        sf_pre_s = eval_context.compute_pre_sf(prefill_pool_id, p_pool, eff_hit)
        denom_pre = max(_RATE_EPS, sf_pre_s)
        pp_bubble_factor = (
            1.0 + pp_bubble_stages / float(num_chunks)
            if pp_bubble_stages > 0.0
            else 1.0
        )
        pre_bub_s = sf_pre_s * pp_bubble_factor
        for item in group_items:
            pcie_s = (
                float(scenario.isl)
                * max(0.0, eff_hit - hit_rate * item.hbm_ratio)
                * item.pcie_slope
                if item.pcie_slope > 0.0
                else 0.0
            )
            rate_p_eff = item.prefill_dp_instances_f / max(
                _RATE_EPS, denom_pre + pcie_s
            )
            xfer_s = (wire_b_h * item.inv_min_c_dcn_ms) * UNIT_MILLI
            rate_dcn_cap = item.dcn_scale / xfer_s if xfer_s > 0.0 else math.inf
            coupled_cap = min(rate_p_eff, item.rate_d, rate_dcn_cap)
            ttft_service_s = pre_bub_s + pcie_s + xfer_s
            if eval_context.think_s is None:
                coupled = coupled_cap
                ll_ttft_s = max(
                    0.0,
                    item.conc_f / max(_RATE_EPS, coupled) - item.dec_dur_s,
                )
            else:
                coupled = min(
                    coupled_cap,
                    item.conc_f
                    / max(
                        _RATE_EPS,
                        ttft_service_s + item.dec_dur_s + eval_context.think_s,
                    ),
                )
                ll_ttft_s = max(
                    0.0,
                    item.conc_f / max(_RATE_EPS, coupled)
                    - item.dec_dur_s
                    - eval_context.think_s,
                )
            ttft_h = max(
                ttft_service_s * UNIT_MILLIS_PER_SECOND + item.queue_wait_d_ms,
                ll_ttft_s * UNIT_MILLIS_PER_SECOND,
            )
            tput = coupled * float(scenario.isl + scenario.osl) * item.inv_chips
            slo_ok = item.tpot_ok and (
                eval_context.ttft_slo is None or ttft_h <= eval_context.ttft_slo
            )
            _record_grid_candidate(
                state,
                (tput, item.inter_d, item.neg_chips, item.point),
                slo_ok=slo_ok,
                sort_by=eval_context.sort_by,
            )


def _layout_sig_tuple(
    point: ParetoPoint,
) -> tuple[tuple[object, ...], tuple[object, ...] | None, int]:
    """Return structural layout signature matching solver diversity rules."""
    return (
        _pool_sig(point.decode_pool),
        _pool_sig(point.prefill_pool) if point.prefill_pool else None,
        point.total_chips,
    )


def _derive_winner_range_and_crossover(
    scenario: TrafficScenario,
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
    if plan.recommended is None or not winners:
        return (None, None)
    base_hit_rate = round(float(scenario.prefix_hit_rate), _HIT_ROUND_DIGITS)
    base_idx = min(
        range(len(winners)), key=lambda i: abs(winners[i][0] - base_hit_rate)
    )
    base_sig = _layout_sig_tuple(plan.recommended)
    sigs = [
        (hit_rate, (d_sig, p_sig, chips))
        for hit_rate, (d_sig, p_sig, chips, _, _) in winners
    ]
    win_range: tuple[float, float] | None
    if sigs[base_idx][1] != base_sig:
        win_range = None
    else:
        lo_idx = base_idx
        while lo_idx > 0 and sigs[lo_idx - 1][1] == base_sig:
            lo_idx -= 1
        hi_idx = base_idx
        while hi_idx + 1 < len(sigs) and sigs[hi_idx + 1][1] == base_sig:
            hi_idx += 1
        win_range = (sigs[lo_idx][0], sigs[hi_idx][0])

    cross: float | None = None
    best_dist = math.inf
    for idx in range(len(sigs) - 1):
        hit_a, sig_a = sigs[idx]
        hit_b, sig_b = sigs[idx + 1]
        if sig_a != sig_b:
            mid = round(0.5 * (hit_a + hit_b), _HIT_ROUND_DIGITS)
            dist = abs(mid - base_hit_rate)
            if dist < best_dist:
                best_dist = dist
                cross = mid
    return (win_range, cross)


class _CrossoverOpts(TypedDict, total=False):
    """Keyword options for heuristic crossover planning."""

    activation_model: ExpertActivationModel | None
    collective_latency_s: float
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None
    sort_by: SortBy


class _MixStepOpts(TypedDict, total=False):
    """Keyword options for mixed step serial_estimate_s evaluation."""

    prefill_tokens: int
    prefix_hit_rate: float
    shared_prefix_tokens: int


class _SweepKwargs(TypedDict, total=False):
    """Optional keyword arguments for prefix-hit crossover sweep."""

    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None


class _CrossoverPlanner:
    """Precompute candidate pool metadata and heuristic serial_estimate_s caches."""

    def __init__(
        self,
        geometry: ModelGeometry,
        dtypes: Dtypes,
        chip: ChipSpec,
        scenario: TrafficScenario,
        **opts: Unpack[_CrossoverOpts],
    ) -> None:
        """Initialize cache containers and compute static parameters."""
        self.geometry = geometry
        self.dtypes = dtypes
        self.chip = chip
        self.scenario = scenario
        self.activation_model = opts.get("activation_model") or ExpertActivationModel()
        self.collective_latency_s = float(opts.get("collective_latency_s", 0.0))
        self.sort_by: SortBy = opts.get("sort_by", "throughput")
        self.topo_map = _resolve_topo_map(chip, opts.get("topologies"))
        self.ctx_mid: int = scenario.isl + scenario.osl // 2
        self.shared_prefix_tokens: int = shared_prefix_tokens(scenario)
        self.effective_accept_len = max(1.0, scenario.accept_len)
        self.spec_k = resolve_spec_k(geometry, scenario)
        self.dec_steps = float(max(0, scenario.osl - 1)) / self.effective_accept_len
        has_dcn = chip.dcn_bw > 0.0 and math.isfinite(chip.dcn_bw)
        self.think_s: float | None = scenario.think_time_s
        self.need_wire = has_dcn

        self.pool_ids: dict[tuple[object, ...], int] = {}
        self.pool_plans: list[
            tuple[SliceTopology | None, Parallel, tuple[int, ...] | None]
        ] = []
        self.pre_step_cache: dict[tuple[int, int, int, int], float] = {}
        self.dec_cache: dict[tuple[int, int], tuple[float, float]] = {}
        self.mix_cache: dict[tuple[int, int, int, float, int], float] = {}
        self.pure_chunk_cache: dict[tuple[int, int, float], float] = {}
        self.kv_b_cache: dict[
            tuple[int, int, int, int, str, tuple[int, ...] | None, int],
            tuple[float, float],
        ] = {}
        self._init_wire()

    def _init_wire(self) -> None:
        """Initialize unpadded KV wire transfer bytes when DCN is enabled."""
        geometry = self.geometry
        self.sw_w = geometry.sliding_window
        if not self.need_wire:
            self.wire_state_b = 0.0
            self.wire_kv_step1 = 0.0
            self.wire_kv_at_sw = 0.0
            self.wire_kv_post_sw = 0.0
            return
        par_w = Parallel(tp=1, pp=1, ranks_per_chip=self.chip.cores_per_chip)
        bd_w0 = hbm_breakdown_per_core_bytes(
            geometry,
            self.dtypes,
            par_w,
            SeqKvLayout(1, 0, physical_kv_layout=False),
            chip=self.chip,
        )
        bd_w1 = hbm_breakdown_per_core_bytes(
            geometry,
            self.dtypes,
            par_w,
            SeqKvLayout(1, 1, physical_kv_layout=False),
            chip=self.chip,
        )
        self.wire_state_b, self.wire_kv_step1 = bd_w0.state, bd_w1.kv
        if self.sw_w > 0:
            bd_w_sw = hbm_breakdown_per_core_bytes(
                geometry,
                self.dtypes,
                par_w,
                SeqKvLayout(1, self.sw_w, physical_kv_layout=False),
                chip=self.chip,
            )
            bd_w_sw1 = hbm_breakdown_per_core_bytes(
                geometry,
                self.dtypes,
                par_w,
                SeqKvLayout(1, self.sw_w + 1, physical_kv_layout=False),
                chip=self.chip,
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

    def get_b_kv_parts(self, pool: PoolMeshPlan, ctx_len: int) -> tuple[float, float]:
        """Calculate linear and fixed-window KV cache bytes per core for a pool."""
        kv_cache_key = (
            pool.tp,
            pool.pp,
            pool.pcp,
            pool.kv_split,
            pool.residual,
            pool.pp_layer_partition,
            ctx_len,
        )
        cached_value = self.kv_b_cache.get(kv_cache_key)
        if cached_value is None:
            cached_value = _kv_parts_for_pool(
                self.geometry, self.dtypes, self.chip, pool, ctx_len
            )
            self.kv_b_cache[kv_cache_key] = cached_value
        return cached_value

    def get_pool_id(self, pool: PoolMeshPlan) -> int:
        """Allocate or look up cached index for a PoolMeshPlan."""
        pool_key: tuple[object, ...] = (
            *_pool_sig(pool),
            pool.residual,
            pool.mesh_axes,
            pool.pp_layer_partition,
        )
        pool_id = self.pool_ids.get(pool_key)
        if pool_id is None:
            topology = self.topo_map.get(pool.topology)
            if topology is None:
                known = ", ".join(sorted(self.topo_map))
                msg = f"Unknown topology {pool.topology!r}; known topologies: {known}"
                raise ValueError(msg)
            pool_id = len(self.pool_plans)
            self.pool_ids[pool_key] = pool_id
            parallel = _pool_parallel(
                pool, topology, default_ranks=self.chip.cores_per_chip
            )
            uneven_part = _resolve_pp_partition(pool, self.geometry.num_layers)
            self.pool_plans.append((topology, parallel, uneven_part))
        return pool_id

    def _eval_stage_or_step_floor(
        self,
        parallel: Parallel,
        shape: StepShape,
        opts: FloorOptions,
        pp_part: tuple[int, ...] | None,
    ) -> StepFloor:
        """Evaluate step_floor or exact per-stage partition floor for uneven PP."""
        if parallel.pp > 1 and pp_part:
            return scale_pp_stage_floor(
                self.geometry, self.chip, parallel, (shape, opts), pp_part
            )
        return step_floor(self.geometry, self.chip, parallel, shape, opts)

    def compute_pre_sf(
        self, pool_id: int, pool: PoolMeshPlan, hit_rate: float
    ) -> float:
        """Compute unbubbled prefill step floor latency under prefix caching."""
        h_clamped = max(0.0, min(1.0, hit_rate))
        eff_tok = (
            max(1, math.ceil((1.0 - h_clamped) * self.scenario.isl))
            if self.scenario.isl > 0
            else 0
        )
        eff_pre = (
            max(
                1,
                math.ceil(
                    self.scenario.isl * max(0.0, 1.0 - h_clamped) / max(1, pool.pcp)
                ),
            )
            if self.scenario.isl > 0
            else 0
        )
        cached_pre = (
            round(self.scenario.isl * h_clamped) if self.scenario.isl > 0 else 0
        )
        step_cache_key = (pool_id, eff_tok, eff_pre, cached_pre)
        step_floor_s = self.pre_step_cache.get(step_cache_key)
        if step_floor_s is None:
            topology, parallel, pp_part = self.pool_plans[pool_id]
            shape = StepShape(
                batch=0,
                rows_per_seq=0,
                context=0,
                state_writes=0,
                prefill_tokens=self.scenario.isl,
                prefix_hit_rate=h_clamped,
            )
            opts = FloorOptions(
                dtypes=self.dtypes,
                collective_latency_s=self.collective_latency_s,
                topology=topology,
                activation_model=self.activation_model,
            )
            sf_obj = self._eval_stage_or_step_floor(parallel, shape, opts, pp_part)
            step_floor_s = sf_obj.serial_estimate_s
            self.pre_step_cache[step_cache_key] = step_floor_s
        return step_floor_s

    def _eval_pp_serial_step(
        self,
        parallel: Parallel,
        shape: StepShape,
        opts: FloorOptions,
        pp_part: tuple[int, ...] | None,
        *,
        apply_ep: bool,
    ) -> float:
        """Evaluate PP-aware serial cycle duration for one decode or verify step."""
        batch = max(1, shape.batch)
        pp = parallel.pp
        rows_per_seq = max(1, shape.rows_per_seq)
        if pp <= 1:
            step_res = step_floor(self.geometry, self.chip, parallel, shape, opts)
            if not apply_ep:
                return step_res.serial_estimate_s
            return scaled_ep_serial_s(
                step_res,
                self.chip,
                self.geometry,
                parallel,
                batch * rows_per_seq * max(1, parallel.dp),
            )
        num_microbatches = min(pp, batch)
        micro_batch = math.ceil(batch / num_microbatches)
        mb_shape = replace(shape, batch=micro_batch)
        mb_tokens = micro_batch * rows_per_seq * max(1, parallel.dp)
        if pp_part is not None and len(pp_part) == pp:
            sf_max = scale_pp_stage_floor(
                self.geometry,
                self.chip,
                parallel,
                (mb_shape, opts),
                pp_part,
                reduce="max",
            )
            sf_sum = scale_pp_stage_floor(
                self.geometry,
                self.chip,
                parallel,
                (mb_shape, opts),
                pp_part,
                reduce="sum",
            )
            if apply_ep:
                stage_s = scaled_ep_serial_s(
                    sf_max, self.chip, self.geometry, parallel, mb_tokens
                )
                ep_delta = max(0.0, stage_s - sf_max.serial_estimate_s)
                sum_s = sf_sum.serial_estimate_s + float(pp) * ep_delta
            else:
                stage_s = sf_max.serial_estimate_s
                sum_s = sf_sum.serial_estimate_s
            return max(sum_s, float(num_microbatches) * stage_s)
        stage_sf = step_floor(self.geometry, self.chip, parallel, mb_shape, opts)
        stage_s = (
            scaled_ep_serial_s(stage_sf, self.chip, self.geometry, parallel, mb_tokens)
            if apply_ep
            else stage_sf.serial_estimate_s
        )
        if batch < pp:
            full_sf = step_floor(
                self.geometry,
                self.chip,
                replace(parallel, pp=1),
                mb_shape,
                opts,
            )
            full_s = (
                scaled_ep_serial_s(
                    full_sf, self.chip, self.geometry, parallel, mb_tokens
                )
                if apply_ep
                else full_sf.serial_estimate_s
            )
            return max(full_s, float(num_microbatches) * stage_s)
        return float(pp) * stage_s

    def _eval_spec_dec_sf(
        self,
        pool_id: int,
        resident_batch: int,
    ) -> tuple[float, float]:
        """Compute speculative cycle and draft delta seconds for decode batch."""
        topology, parallel, pp_part = self.pool_plans[pool_id]
        opts = FloorOptions(
            dtypes=self.dtypes,
            collective_latency_s=self.collective_latency_s,
            topology=topology,
            activation_model=self.activation_model,
        )
        batch = max(1, resident_batch)
        pp = parallel.pp
        num_microbatches = min(pp, batch) if pp > 1 else 1
        micro_batch = math.ceil(batch / num_microbatches) if pp > 1 else batch
        spec_shape = SpecShape(
            batch=micro_batch,
            context=self.ctx_mid,
            spec_k=self.spec_k,
            shared_prefix_tokens=self.shared_prefix_tokens,
        )
        stage_spec_sf = spec_cycle_floor(
            self.geometry, self.chip, parallel, spec_shape, opts
        )
        mb_tokens = micro_batch * (self.spec_k + 1) * max(1, parallel.dp)
        stage_spec_s = scaled_ep_serial_s(
            stage_spec_sf, self.chip, self.geometry, parallel, mb_tokens
        )
        if pp <= 1:
            total_serial_s = stage_spec_s
        elif batch < pp:
            full_spec_sf = spec_cycle_floor(
                self.geometry,
                self.chip,
                replace(parallel, pp=1),
                spec_shape,
                opts,
            )
            full_spec_s = scaled_ep_serial_s(
                full_spec_sf, self.chip, self.geometry, parallel, mb_tokens
            )
            total_serial_s = max(full_spec_s, float(num_microbatches) * stage_spec_s)
        else:
            total_serial_s = float(pp) * stage_spec_s
        ver_shape = StepShape(
            batch=batch,
            rows_per_seq=self.spec_k + 1,
            context=self.ctx_mid,
            state_writes=1,
            shared_prefix_tokens=self.shared_prefix_tokens,
        )
        verify_serial_s = self._eval_pp_serial_step(
            parallel,
            ver_shape,
            opts,
            pp_part if pp > 1 else None,
            apply_ep=True,
        )
        if pp > 1 and pp_part is not None and len(pp_part) == pp:
            draft_mods = tuple(
                module
                for module in stage_spec_sf.modules
                if module.name.startswith("draft_")
            )
            draft_sf = combine_modules(
                draft_mods,
                self.chip,
                self.collective_latency_s,
                topology=topology,
                parallel=parallel,
            )
            draft_s = scaled_ep_serial_s(
                draft_sf, self.chip, self.geometry, parallel, mb_tokens
            ) * float(num_microbatches)
            return (verify_serial_s + draft_s, max(0.0, draft_s))
        draft_s = max(0.0, total_serial_s - verify_serial_s)
        return (total_serial_s, draft_s)

    def get_dec_sf(self, pool_id: int, resident_batch: int) -> tuple[float, float]:
        """Retrieve or compute PP-aware decode step serial_estimate_s latency."""
        key = (pool_id, resident_batch)
        cached_value = self.dec_cache.get(key)
        if cached_value is not None:
            return cached_value
        if self.spec_k > 0:
            cached_value = self._eval_spec_dec_sf(pool_id, resident_batch)
            self.dec_cache[key] = cached_value
            return cached_value
        topology, parallel, pp_part = self.pool_plans[pool_id]
        shape = StepShape(
            batch=resident_batch,
            rows_per_seq=1,
            context=self.ctx_mid,
            shared_prefix_tokens=self.shared_prefix_tokens,
        )
        opts = FloorOptions(
            dtypes=self.dtypes,
            collective_latency_s=self.collective_latency_s,
            topology=topology,
            activation_model=self.activation_model,
        )
        serial_s = self._eval_pp_serial_step(
            parallel, shape, opts, pp_part, apply_ep=True
        )
        cached_value = (serial_s, 0.0)
        self.dec_cache[key] = cached_value
        return cached_value

    def compute_mix_sf(
        self,
        pool_id: int,
        resident_batch: int,
        chunk_tok: int,
        **opts: Unpack[_MixStepOpts],
    ) -> float:
        """Compute mixed prefill-decode step serial_estimate_s latency."""
        prefill_tokens = opts.get("prefill_tokens", chunk_tok)
        prefix_hit_rate = opts.get("prefix_hit_rate", 0.0)
        shared_tok = opts.get("shared_prefix_tokens", self.shared_prefix_tokens)
        key = (pool_id, resident_batch, prefill_tokens, prefix_hit_rate, shared_tok)
        cached_value = self.mix_cache.get(key)
        if cached_value is None:
            topology, parallel, pp_part = self.pool_plans[pool_id]
            shape = StepShape(
                batch=resident_batch,
                rows_per_seq=self.spec_k + 1,
                context=self.ctx_mid,
                state_writes=1,
                prefill_tokens=prefill_tokens,
                prefix_hit_rate=prefix_hit_rate,
                shared_prefix_tokens=shared_tok,
            )
            floor_opts = FloorOptions(
                dtypes=self.dtypes,
                collective_latency_s=self.collective_latency_s,
                topology=topology,
                activation_model=self.activation_model,
            )
            cached_value = self._eval_pp_serial_step(
                parallel, shape, floor_opts, pp_part, apply_ep=False
            )
            self.mix_cache[key] = cached_value
        return cached_value

    def compute_pure_chunk_sf(
        self,
        pool_id: int,
        prefill_tokens: int,
        prefix_hit_rate: float,
    ) -> float:
        """Compute pure-chunk prefill step serial_estimate_s latency for tail work."""
        key = (pool_id, prefill_tokens, prefix_hit_rate)
        cached_value = self.pure_chunk_cache.get(key)
        if cached_value is None:
            topology, parallel, pp_part = self.pool_plans[pool_id]
            shape = StepShape(
                batch=0,
                rows_per_seq=0,
                context=0,
                state_writes=0,
                prefill_tokens=prefill_tokens,
                prefix_hit_rate=prefix_hit_rate,
            )
            opts = FloorOptions(
                dtypes=self.dtypes,
                collective_latency_s=self.collective_latency_s,
                topology=topology,
                activation_model=self.activation_model,
            )
            step_floor_val = self._eval_stage_or_step_floor(
                parallel, shape, opts, pp_part
            )
            cached_value = step_floor_val.serial_estimate_s
            self.pure_chunk_cache[key] = cached_value
        return cached_value

    def add_mixed_cand(
        self,
        point: ParetoPoint,
        decode_pool_id: int,
        mix_meta: list[MixedCandidateMeta],
    ) -> None:
        """Append precomputed MixedCandidateMeta for one mixed candidate."""
        d_pool = point.decode_pool
        decode_dp_instances = max(1, d_pool.instances * max(1, d_pool.dp))
        local_batch = max(1, math.ceil(point.concurrency / decode_dp_instances))
        resident_batch = (
            d_pool.resident_batch_cap
            if d_pool.resident_batch_cap is not None
            else local_batch
        )
        req = _SweepKvRequest(
            pool=d_pool,
            working_set_batch=local_batch,
            ctx_len=max(1, self.scenario.isl + self.scenario.osl),
            b_res_fixed=resident_batch,
            kv_cap_tokens=point.kv_cache_capacity_tokens,
        )
        cap_ratio, hbm_ratio, pcie_k = _calc_sweep_kv_ratios(
            req, self.scenario, self.get_b_kv_parts
        )
        pcie_core_bw = self.chip.pcie_bw / float(max(1, self.chip.cores_per_chip))
        has_pcie = pcie_k > 0.0 and pcie_core_bw > 0.0 and math.isfinite(pcie_core_bw)
        mix_meta.append(
            MixedCandidateMeta(
                point=point,
                d_pool=d_pool,
                decode_pool_id=decode_pool_id,
                resident_batch=resident_batch,
                cap_ratio=cap_ratio,
                hbm_ratio=hbm_ratio,
                pcie_slope=(pcie_k / pcie_core_bw) if has_pcie else 0.0,
                res_cap=float(resident_batch * decode_dp_instances),
                conc_f=float(point.concurrency),
                inv_chips=1.0 / float(point.total_chips),
                neg_chips=-point.total_chips,
            )
        )

    def add_disagg_cand(
        self,
        point: ParetoPoint,
        decode_pool_id: int,
        dis_ttft_groups: dict[
            tuple[int, float], tuple[PoolMeshPlan, float, list[DisaggTtftItem]]
        ],
    ) -> None:
        """Append precomputed disaggregated candidate entry for crossover sweep."""
        p_pool = point.prefill_pool
        if p_pool is None:
            msg = "Disaggregated point missing prefill_pool"
            raise RuntimeError(msg)
        d_pool = point.decode_pool
        decode_dp_instances = max(1, d_pool.instances * max(1, d_pool.dp))
        local_batch = max(1, math.ceil(point.concurrency / decode_dp_instances))
        resident_batch = (
            d_pool.resident_batch_cap
            if d_pool.resident_batch_cap is not None
            else local_batch
        )
        prefill_pool_id = self.get_pool_id(p_pool)
        prefill_dp_instances = max(1, p_pool.instances * max(1, p_pool.dp))
        working_set_prefill = max(
            1, math.ceil(point.concurrency / prefill_dp_instances)
        )
        req = _SweepKvRequest(
            pool=p_pool,
            working_set_batch=working_set_prefill,
            ctx_len=max(1, self.scenario.isl),
            b_res_fixed=1,
            kv_cap_tokens=None,
        )
        cap_ratio, hbm_ratio, pcie_k = _calc_sweep_kv_ratios(
            req, self.scenario, self.get_b_kv_parts
        )
        sf_dec_s, _draft_s = self.get_dec_sf(decode_pool_id, resident_batch)
        inv_al_ms = UNIT_MILLIS_PER_SECOND / self.effective_accept_len
        dec_dur_s = self.dec_steps * sf_dec_s
        conc_f = float(point.concurrency)
        res_cap_d = float(resident_batch * decode_dp_instances)
        rate_d = min(conc_f, res_cap_d) / max(_RATE_EPS, dec_dur_s)
        c_demand_d = min(conc_f, float(local_batch * decode_dp_instances))
        queue_wait_d_ms = (
            max(0.0, (c_demand_d / max(_RATE_EPS, res_cap_d)) - 1.0)
            * dec_dur_s
            * UNIT_MILLIS_PER_SECOND
        )
        tpot_d = sf_dec_s * inv_al_ms
        inter_d = UNIT_MILLIS_PER_SECOND / max(_TPOT_EPS, tpot_d)
        tpot_ok = (
            self.scenario.tpot_slo_ms is None or tpot_d <= self.scenario.tpot_slo_ms
        )
        group_key = (prefill_pool_id, cap_ratio)
        pcie_core_bw = self.chip.pcie_bw / float(max(1, self.chip.cores_per_chip))
        has_pcie = pcie_k > 0.0 and pcie_core_bw > 0.0 and math.isfinite(pcie_core_bw)
        has_dcn = self.chip.dcn_bw > 0.0 and math.isfinite(self.chip.dcn_bw)
        min_c = float(max(1, min(p_pool.chips_per_instance, d_pool.chips_per_instance)))
        min_pool = float(
            min(
                p_pool.instances * p_pool.chips_per_instance,
                d_pool.instances * d_pool.chips_per_instance,
            )
        )
        inv_min_c_dcn_ms = (
            UNIT_MILLIS_PER_SECOND / (min_c * self.chip.dcn_bw) if has_dcn else 0.0
        )
        dcn_scale = min_pool / max(1.0, min_c)
        dis_ttft_groups.setdefault(group_key, (p_pool, float(p_pool.pp - 1), []))[
            2
        ].append(
            DisaggTtftItem(
                prefill_dp_instances_f=float(prefill_dp_instances),
                rate_d=rate_d,
                inv_chips=1.0 / float(point.total_chips),
                tpot_ok=tpot_ok,
                inter_d=inter_d,
                neg_chips=-point.total_chips,
                pcie_slope=(pcie_k / pcie_core_bw) if has_pcie else 0.0,
                hbm_ratio=hbm_ratio,
                inv_min_c_dcn_ms=inv_min_c_dcn_ms,
                dcn_scale=dcn_scale,
                dec_dur_s=dec_dur_s,
                conc_f=conc_f,
                point=point,
                queue_wait_d_ms=queue_wait_d_ms,
            )
        )


def setup_crossover_sweep(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    cands: Sequence[ParetoPoint],
    **opts: Unpack[_CrossoverOpts],
) -> tuple[CrossoverPools, _GridEvalContext]:
    """Precompute candidate pool metadata and heuristic step floor caches."""
    planner = _CrossoverPlanner(geometry, dtypes, chip, scenario, **opts)
    mix_meta: list[MixedCandidateMeta] = []
    dis_ttft_groups: dict[
        tuple[int, float], tuple[PoolMeshPlan, float, list[DisaggTtftItem]]
    ] = {}

    for point in cands:
        decode_pool_id = planner.get_pool_id(point.decode_pool)
        if point.mode == "mixed":
            planner.add_mixed_cand(point, decode_pool_id, mix_meta)
        else:
            planner.add_disagg_cand(point, decode_pool_id, dis_ttft_groups)

    def _compute_mix_sf(
        pool_id: int,
        resident_batch: int,
        prefill_tokens: int,
        prefix_hit_rate: float,
        chunk_tok: int,
        shared_prefix_tokens: int,
    ) -> float:
        """Forward mean causal context options into compute_mix_sf."""
        return planner.compute_mix_sf(
            pool_id,
            resident_batch,
            chunk_tok,
            prefill_tokens=prefill_tokens,
            prefix_hit_rate=prefix_hit_rate,
            shared_prefix_tokens=shared_prefix_tokens,
        )

    pools = CrossoverPools(
        mix_meta=tuple(mix_meta),
        dis_ttft_groups=dis_ttft_groups,
    )
    eval_context = _GridEvalContext(
        mix_cache=planner.mix_cache,
        pure_chunk_cache=planner.pure_chunk_cache,
        pool_plans=planner.pool_plans,
        compute_pre_sf=planner.compute_pre_sf,
        compute_mix_sf=_compute_mix_sf,
        compute_pure_chunk_sf=planner.compute_pure_chunk_sf,
        get_wire_b=planner.get_wire_b,
        need_wire=planner.need_wire,
        prefill_chunk=scenario.prefill_chunk,
        inv_al_ms=UNIT_MILLIS_PER_SECOND / planner.effective_accept_len,
        dec_steps=planner.dec_steps,
        think_s=planner.think_s,
        ttft_slo=scenario.ttft_slo_ms,
        tpot_slo=scenario.tpot_slo_ms,
        get_dec_sf=planner.get_dec_sf,
        spec_k=planner.spec_k,
        sort_by=planner.sort_by,
    )
    return (pools, eval_context)


_SEED_FRONTIER_LIMIT: Final[int] = 6


def _point_struct_key(point: ParetoPoint) -> tuple[object, ...]:
    """Return total-order layout-and-concurrency identity key for tie-breaking."""
    return (
        _schema_point_struct_key(point),
        point.decode_pool.residual,
        point.decode_pool.moe_dispatch,
        point.prefill_pool.residual if point.prefill_pool else "",
        point.prefill_pool.moe_dispatch if point.prefill_pool else "",
        point.total_chips,
        point.concurrency,
    )


def _eval_bounded_mixed_grid(
    hit_rate: float,
    scenario: TrafficScenario,
    mix_meta: Sequence[MixedCandidateMeta],
    eval_context: _GridEvalContext,
    bounds_state: tuple[CandidateState, set[tuple[object, ...]]],
) -> None:
    """Evaluate mixed candidates using sound pure-decode branch-and-bound pruning."""
    state, seed_keys = bounds_state
    eff_isl_1 = max(1, math.ceil((1.0 - hit_rate) * scenario.isl))
    num_chunks_1 = max(1, math.ceil(eff_isl_1 / eval_context.prefill_chunk))
    defaults = (num_chunks_1, min(eval_context.prefill_chunk, eff_isl_1))
    can_prune_by_tput = eval_context.sort_by == "throughput"
    rest_idxs: list[int] = []
    for idx, meta in enumerate(mix_meta):
        if _point_struct_key(meta.point) in seed_keys:
            cand, slo_ok = _eval_one_mixed_meta(
                meta, hit_rate, scenario, eval_context, defaults
            )
            _record_grid_candidate(
                state, cand, slo_ok=slo_ok, sort_by=eval_context.sort_by
            )
        else:
            rest_idxs.append(idx)
    for idx in rest_idxs:
        meta = mix_meta[idx]
        best_bar = state.max_slo
        if can_prune_by_tput and best_bar > 0.0:
            _, num_chunks, _, _, _ = _mixed_chunk_context_stats(
                hit_rate,
                meta.cap_ratio,
                scenario.isl,
                eval_context.prefill_chunk,
                defaults,
            )
            dec_ub = _mixed_pure_decode_upper_bound(
                meta, num_chunks, scenario, eval_context
            )
            if dec_ub is not None and dec_ub * _TOL_BAND < best_bar:
                continue
        cand, slo_ok = _eval_one_mixed_meta(
            meta, hit_rate, scenario, eval_context, defaults
        )
        _record_grid_candidate(state, cand, slo_ok=slo_ok, sort_by=eval_context.sort_by)


def _eval_grid_h_bounded(
    hit_rate: float,
    scenario: TrafficScenario,
    pools: CrossoverPools,
    eval_context: _GridEvalContext,
    seed_keys: set[tuple[object, ...]],
) -> (
    tuple[
        float,
        tuple[tuple[object, ...], tuple[object, ...] | None, int, float, ParetoPoint],
    ]
    | None
):
    """Find winning candidate at hit rate with bit-identical sound pruning."""
    state = CandidateState()
    _eval_grid_disagg_ttft(
        hit_rate, scenario, pools.dis_ttft_groups, eval_context, state
    )
    _eval_bounded_mixed_grid(
        hit_rate, scenario, pools.mix_meta, eval_context, (state, seed_keys)
    )
    sort_by = eval_context.sort_by
    candidates = (
        [
            cand
            for cand in state.near_slo
            if _primary_metric(cand, sort_by) * _TOL_BAND >= state.max_slo
        ]
        if state.near_slo
        else [
            cand
            for cand in state.near_all
            if _primary_metric(cand, sort_by) * _TOL_BAND >= state.max_all
        ]
    )
    if not candidates:
        return None
    winner = _best_frontier_winner(candidates, sort_by=sort_by)
    return (
        hit_rate,
        (
            _pool_sig(winner.decode_pool),
            _pool_sig(winner.prefill_pool) if winner.prefill_pool else None,
            winner.total_chips,
            winner.token_throughput_per_chip,
            winner,
        ),
    )


def sweep_prefix_hit_crossovers(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    plan: ServingPlan,
    **opts: Unpack[_SweepKwargs],
) -> tuple[tuple[float, float] | None, float | None]:
    """Run a heuristic serial_estimate_s sensitivity sweep across prefix hit rates."""
    if plan.recommended is None:
        return (None, None)
    raw_cands = (
        plan.sweep_candidates
        if plan.sweep_candidates
        else (plan.candidates if plan.candidates else plan.frontier)
    )
    if not raw_cands:
        return (None, None)
    scenario = plan.scenario
    cands = (
        raw_cands
        if plan.sweep_candidates
        else prune_sweep_candidates(geometry, dtypes, chip, scenario, raw_cands)
    )
    grid = tuple(
        sorted(
            {
                *(
                    round(float(hit_rate), _HIT_ROUND_DIGITS)
                    for hit_rate in _DEFAULT_HIT_RATES
                ),
                round(scenario.prefix_hit_rate, _HIT_ROUND_DIGITS),
            }
        )
    )
    pools, eval_context = setup_crossover_sweep(
        geometry,
        dtypes,
        chip,
        scenario,
        cands,
        activation_model=plan.activation_model,
        collective_latency_s=plan.collective_latency_s,
        topologies=opts.get("topologies"),
        sort_by=plan.sort_by,
    )
    seed_keys: set[tuple[object, ...]] = {_point_struct_key(plan.recommended)}
    for frontier_point in plan.frontier[:_SEED_FRONTIER_LIMIT]:
        seed_keys.add(_point_struct_key(frontier_point))
    winners_rev: list[
        tuple[
            float,
            tuple[
                tuple[object, ...], tuple[object, ...] | None, int, float, ParetoPoint
            ],
        ]
    ] = []
    for hit_rate in reversed(grid):
        entry = _eval_grid_h_bounded(hit_rate, scenario, pools, eval_context, seed_keys)
        if entry is not None:
            winners_rev.append(entry)
            seed_keys.add(_point_struct_key(entry[1][4]))
    winners = list(reversed(winners_rev))
    if not winners:
        return (None, None)
    return _derive_winner_range_and_crossover(scenario, plan, winners)
