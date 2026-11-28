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
"""Physics-only roofline solver and per-pool layout ordering."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, Literal, TypedDict, Unpack

from .chips import TOPOLOGIES, ChipSpec, SliceTopology, topologies_for_chip
from .collectives import (
    LinkCutLoad,
    eval_link_subset_bound_s,
    submesh_for_group,
)
from .constants import (
    CAUSAL_MIDPOINT_HALF,
    DEFAULT_HBM_RESERVE_GB,
    DEFAULT_MAX_INSTANCES,
    EP_PAIR_RANKS,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    LINEAR_ATTENTION_LAYER_TOKENS,
    MIN_DISAGG_INSTANCES,
    NO_FEASIBLE_LAYOUT_WARNING,
    SLO_VERDICT_ESTIMATE_MEETS,
    SLO_VERDICT_RANK,
    SLO_VERDICT_RULED_OUT,
    SLO_VERDICT_UNDECIDED,
    UNDECIDED_SLO_WARNING,
    UNIT_GIGA,
    UNIT_MILLIS_PER_SECOND,
    UNKNOWN_THINK_TIME_WARNING,
)
from .memory import (
    HbmBreakdown,
    KvFitInput,
    SeqKvLayout,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    kv_bytes_per_core_parts,
    kv_capacity_and_hit_rate,
)
from .modules import Dtypes, Parallel, StepShape
from .occupancy import ExpertActivationModel
from .schema import (
    FloorTimes,
    ParetoPoint,
    PoolFloorRecord,
    PoolMeshPlan,
    ScenarioOverrides,
    SearchStats,
    ServingPlan,
    SortBy,
    TrafficScenario,
    candidate_sort_key,
    effective_slo_verdict,
)
from .step import (
    FloorOptions,
    SpecShape,
    StepFloor,
    UnpaddedFloorOptions,
    spec_cycle_floor,
    step_floor,
    unpadded_strict_resource_buckets,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .constants import SloVerdict
    from .geometry import ModelGeometry

EXAMPLE_SCENARIOS: dict[str, TrafficScenario] = {
    "chat-8k1k": TrafficScenario(
        name="example:chat-8k1k",
        isl=8192,
        osl=1024,
        concurrencies=EXAMPLE_CONCURRENCIES,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "reasoning-1k8k": TrafficScenario(
        name="example:reasoning-1k8k",
        isl=1024,
        osl=8192,
        concurrencies=EXAMPLE_CONCURRENCIES,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "coding-agent": TrafficScenario(
        name="example:coding-agent",
        isl=16384,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128),
        prefix_hit_rate=0.92,
        accept_len=3.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
        spec_tokens=3,
    ),
    "agentic-64k": TrafficScenario(
        name="example:agentic-64k",
        isl=65536,
        osl=1024,
        concurrencies=EXAMPLE_CONCURRENCIES,
        prefix_hit_rate=0.88,
        accept_len=3.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
        spec_tokens=3,
    ),
    "custom": TrafficScenario(
        name="example:custom",
        isl=4096,
        osl=1024,
        concurrencies=(1, 4, 16, 64),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
}


def build_example_scenario(
    preset: str = "chat-8k1k",
    *,
    slo_tier: str = "none",
    overrides: ScenarioOverrides | None = None,
) -> TrafficScenario:
    """Resolve a TrafficScenario from a labeled preset and typed overrides."""
    if preset not in EXAMPLE_SCENARIOS:
        msg = (
            f"Unknown example scenario preset {preset!r}; "
            f"known: {sorted(EXAMPLE_SCENARIOS)}"
        )
        raise ValueError(msg)
    if slo_tier not in EXAMPLE_SLO_TIERS:
        msg = f"Unknown SLO tier {slo_tier!r}; known: {sorted(EXAMPLE_SLO_TIERS)}"
        raise ValueError(msg)
    base = EXAMPLE_SCENARIOS[preset]
    overrides_val = overrides or ScenarioOverrides()
    tier_ttft, tier_tpot = EXAMPLE_SLO_TIERS[slo_tier]
    return replace(
        base,
        isl=base.isl if overrides_val.isl is None else overrides_val.isl,
        osl=base.osl if overrides_val.osl is None else overrides_val.osl,
        concurrencies=(
            base.concurrencies
            if overrides_val.concurrencies is None
            else overrides_val.concurrencies
        ),
        prefix_hit_rate=(
            base.prefix_hit_rate
            if overrides_val.prefix_hit_rate is None
            else overrides_val.prefix_hit_rate
        ),
        accept_len=base.accept_len
        if overrides_val.accept_len is None
        else overrides_val.accept_len,
        think_time_s=(
            base.think_time_s
            if overrides_val.think_time_s is None
            else overrides_val.think_time_s
        ),
        prefill_chunk=(
            base.prefill_chunk
            if overrides_val.prefill_chunk is None
            else overrides_val.prefill_chunk
        ),
        ttft_slo_ms=(
            overrides_val.ttft_slo_ms
            if overrides_val.ttft_slo_ms is not None
            else (tier_ttft if tier_ttft is not None else base.ttft_slo_ms)
        ),
        tpot_slo_ms=(
            overrides_val.tpot_slo_ms
            if overrides_val.tpot_slo_ms is not None
            else (tier_tpot if tier_tpot is not None else base.tpot_slo_ms)
        ),
        kv_working_set_turns=(
            base.kv_working_set_turns
            if overrides_val.kv_working_set_turns is None
            else overrides_val.kv_working_set_turns
        ),
    )


@dataclass(frozen=True, slots=True)
class SolveOptions:
    """Search mode, slice budget, HBM reserve, pin, and ordering options."""

    mode: Literal["auto", "mixed", "disagg"] = "auto"
    max_chips: int | None = None
    max_instances: int | None = None
    hbm_reserve_gb: float = DEFAULT_HBM_RESERVE_GB
    topologies: tuple[SliceTopology, ...] | None = None
    collective_latency_s: float = 0.0
    sort_by: SortBy = "throughput"
    include_pairs: bool = False
    top_k: int | None = None
    prefill_topology: str | None = None
    decode_topology: str | None = None
    pd_ratio: tuple[int, int] | None = None
    activation_model: ExpertActivationModel = field(
        default_factory=ExpertActivationModel
    )

    def __post_init__(self) -> None:
        """Validate non-negative physical constraint overrides."""
        if self.hbm_reserve_gb < 0.0:
            msg = f"hbm_reserve_gb must be >= 0, got {self.hbm_reserve_gb}"
            raise ValueError(msg)


def _divisors(n: int) -> tuple[int, ...]:
    """Return all integer divisors of n in ascending order."""
    return tuple(d for d in range(1, n + 1) if n % d == 0)


def _decode_tokens(osl: int) -> int:
    """Return decode-phase output tokens excluding the first prefill token."""
    return max(0, osl - 1)


def _decode_steps_per_req(osl: int, effective_accept_len: float) -> float:
    """Return expected decode steps per request."""
    return float(_decode_tokens(osl)) / max(1.0, effective_accept_len)


def _is_sp_residual_legal(geometry: ModelGeometry, parallel: Parallel) -> bool:
    """Return whether sequence-parallel residual sharding is legal for this pool."""
    return (
        parallel.tp > 1
        and parallel.pp == 1
        and (geometry.num_experts == 0 or parallel.moe == "ep")
    )


def _candidate_residual_parallels(
    geometry: ModelGeometry,
    parallel: Parallel,
) -> tuple[Parallel, ...]:
    """Return candidate residual modes (replicated, optional sp) for one base pool."""
    if _is_sp_residual_legal(geometry, parallel):
        return (parallel, replace(parallel, residual="sp"))
    return (parallel,)


def _valid_head_tp(geometry: ModelGeometry, tp: int) -> bool:
    """Check physical attention head and KV head divisibility for TP degree."""
    if geometry.num_heads > 0 and (
        tp > geometry.num_heads or geometry.num_heads % tp != 0
    ):
        return False
    if geometry.num_kv_heads <= 0 or geometry.kv_lora_rank > 0:
        return True
    return geometry.num_kv_heads % tp == 0 or tp % geometry.num_kv_heads == 0


def _has_recurrent_layers(geometry: ModelGeometry) -> bool:
    """Check whether any layer type in geometry uses recurrent linear attention."""
    return any(tok in LINEAR_ATTENTION_LAYER_TOKENS for tok in geometry.layer_types)


def _valid_gdn_degree(geometry: ModelGeometry, deg: int) -> bool:
    """Check physical recurrent head divisibility along TP degree."""
    if deg <= 1 or not _has_recurrent_layers(geometry):
        return True
    if geometry.linear_v_heads <= 0 or geometry.linear_k_heads <= 0:
        return True
    if geometry.linear_v_heads % deg != 0:
        return False
    unique_k_heads = min(geometry.linear_k_heads, deg)
    if geometry.linear_k_heads % unique_k_heads != 0 or deg % unique_k_heads != 0:
        return False
    return (geometry.linear_v_heads // deg) % (
        geometry.linear_k_heads // unique_k_heads
    ) == 0


def _kv_splits_for_tp(geometry: ModelGeometry, tp: int) -> tuple[int, ...]:
    """Return all physical intra-KV-head-group context split degrees."""
    has_kv = geometry.num_kv_heads > 0 or geometry.kv_lora_rank > 0
    if not has_kv:
        return (1,)
    kv_group = (
        max(1, tp // geometry.num_kv_heads)
        if geometry.num_kv_heads > 0 and geometry.kv_lora_rank == 0
        else tp
    )
    return _divisors(kv_group) if kv_group > 1 else (1,)


def _candidate_pool_moe_modes(
    geometry: ModelGeometry,
    tp: int,
    ep_domain: int,
) -> list[tuple[Literal["tp_slice", "ep"], int]]:
    """Enumerate all physically legal (moe_mode, ep_degree) pairs for a pool."""
    if geometry.num_experts == 0:
        return [("tp_slice", 1)]
    modes: list[tuple[Literal["tp_slice", "ep"], int]] = []
    if tp == 1 or geometry.moe_inter <= 0 or geometry.moe_inter % tp == 0:
        modes.append(("tp_slice", 1))
    for ep in _divisors(ep_domain):
        if (
            ep < EP_PAIR_RANKS
            or ep > geometry.num_experts
            or geometry.num_experts % ep != 0
        ):
            continue
        if ep < tp:
            if tp % ep != 0:
                continue
            tp_sub = tp // ep
            if geometry.moe_inter > 0 and geometry.moe_inter % tp_sub != 0:
                continue
        modes.append(("ep", ep))
    return modes


def _append_axis_parallels(
    geometry: ModelGeometry,
    topology: SliceTopology,
    tp: int,
    kv_splits: tuple[int, ...],
    out: list[Parallel],
) -> None:
    """Append all physically legal (dp, pp, pcp, kv_split, moe) layouts for one TP."""
    mesh_ax = submesh_for_group(tp, topology, topology.cores_per_chip)
    rem_tp = topology.num_cores // tp
    for dp in _divisors(rem_tp):
        rem_dp = rem_tp // dp
        for pp in _divisors(rem_dp):
            if pp > geometry.num_layers:
                continue
            pcp = rem_dp // pp
            moe_opts = _candidate_pool_moe_modes(geometry, tp, tp * dp * pcp)
            out.extend(
                Parallel(
                    tp=tp,
                    kv_split=kvs,
                    moe=moe_mode,
                    moe_dispatch=(
                        "a2a_push"
                        if (moe_mode == "ep" and ep_val > tp)
                        else "allreduce"
                    ),
                    ranks_per_chip=topology.cores_per_chip,
                    dp=dp,
                    ep=ep_val,
                    pp=pp,
                    pcp=pcp,
                    mesh_axes=mesh_ax,
                )
                for kvs in kv_splits
                for moe_mode, ep_val in moe_opts
            )


def _candidate_pool_parallels(
    geometry: ModelGeometry,
    topology: SliceTopology,
) -> list[Parallel]:
    """Enumerate all physically legal single-pool parallelisms on a slice topology."""
    out: list[Parallel] = []
    for tp in _divisors(topology.num_cores):
        if not _valid_head_tp(geometry, tp) or not _valid_gdn_degree(geometry, tp):
            continue
        kv_splits = _kv_splits_for_tp(geometry, tp)
        _append_axis_parallels(geometry, topology, tp, kv_splits, out)
    return out


def _candidate_decode_parallels(
    geometry: ModelGeometry,
    topology: SliceTopology,
) -> list[Parallel]:
    """Enumerate all physically legal decode parallelism layouts (pcp == 1)."""
    return [
        parallel
        for parallel in _candidate_pool_parallels(geometry, topology)
        if parallel.pcp == 1
    ]


def _candidate_prefill_parallels(
    geometry: ModelGeometry,
    topology: SliceTopology,
) -> list[Parallel]:
    """Enumerate all physically legal prefill parallelism layouts for a topology."""
    return _candidate_pool_parallels(geometry, topology)


def _pareto_flags_2d(xs: Sequence[float], ys: Sequence[float]) -> list[bool]:
    """Return non-dominated flags for maximizing (x, y) in O(N log N)."""
    n = len(xs)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (-xs[i], -ys[i]))
    flags = [False] * n
    best_y_strict = -math.inf
    group_max_y = -math.inf
    prev_x: float | None = None
    for idx in order:
        x = xs[idx]
        y = ys[idx]
        if prev_x is None or x != prev_x:
            if prev_x is not None:
                best_y_strict = max(best_y_strict, group_max_y)
            prev_x = x
            group_max_y = y
        if y > best_y_strict and y == group_max_y:
            flags[idx] = True
    return flags


def _pool_tie_key(pool: PoolMeshPlan | None) -> tuple[object, ...]:
    """Build deterministic structural tie-break key for one PoolMeshPlan."""
    if pool is None:
        return ("", 0, 0, 0, 0, 0, 0, 0)
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
    )


def _candidate_sort_key(
    point: ParetoPoint, sort_by: SortBy
) -> tuple[int, float, float, float, int, str]:
    """Delegate candidate ordering to the unified schema sort key."""
    return candidate_sort_key(point, sort_by)


def _order_and_mark_candidates(
    points: Sequence[ParetoPoint],
    sort_by: SortBy,
) -> tuple[tuple[ParetoPoint, ...], tuple[ParetoPoint, ...]]:
    """Mark Pareto members among SLO-met layouts and order all legal candidates."""
    if not points:
        return (), ()
    by_conc: dict[int, list[ParetoPoint]] = {}
    for point in points:
        by_conc.setdefault(point.concurrency, []).append(point)
    ordered: list[ParetoPoint] = []
    for concurrency in sorted(by_conc):
        group = by_conc[concurrency]
        eligible_idx = [
            idx for idx, point_item in enumerate(group) if point_item.slo_met
        ]
        flags = [False] * len(group)
        if eligible_idx:
            eligible_flags = _pareto_flags_2d(
                [group[idx].token_throughput_per_chip for idx in eligible_idx],
                [group[idx].est_interactivity_tok_s for idx in eligible_idx],
            )
            for idx, flag in zip(eligible_idx, eligible_flags, strict=True):
                flags[idx] = flag
        marked = [
            replace(point_item, is_pareto=flag)
            for point_item, flag in zip(group, flags, strict=True)
        ]
        marked.sort(key=lambda point_item: _candidate_sort_key(point_item, sort_by))
        ordered.extend(marked)
    frontier = tuple(point_item for point_item in ordered if point_item.is_pareto)
    return tuple(ordered), frontier


def _pool_record_sort_key(
    record: PoolFloorRecord, sort_by: SortBy
) -> tuple[object, ...]:
    """Build deterministic per-concurrency sort key for one PoolFloorRecord."""
    speed_metric = (
        record.interactivity_tok_s
        if record.interactivity_tok_s is not None
        else (UNIT_MILLIS_PER_SECOND / max(1e-6, record.ttft_ms or 1e-6))
    )
    primary, secondary = (
        (speed_metric, record.token_throughput_per_chip)
        if sort_by == "interactivity"
        else (record.token_throughput_per_chip, speed_metric)
    )
    return (
        not record.slo_met,
        -primary,
        -secondary,
        record.pool.chips_per_instance * record.pool.instances,
        _pool_tie_key(record.pool),
    )


def _order_and_mark_pool_records(
    records: Sequence[PoolFloorRecord],
    sort_by: SortBy,
) -> tuple[PoolFloorRecord, ...]:
    """Mark Pareto members and order all legal per-pool records per concurrency."""
    if not records:
        return ()
    by_conc: dict[int, list[PoolFloorRecord]] = {}
    for record in records:
        by_conc.setdefault(record.concurrency, []).append(record)
    ordered: list[PoolFloorRecord] = []
    for concurrency in sorted(by_conc):
        group = by_conc[concurrency]
        ys = [
            (
                record_item.interactivity_tok_s
                if record_item.interactivity_tok_s is not None
                else UNIT_MILLIS_PER_SECOND / max(1e-6, record_item.ttft_ms or 1e-6)
            )
            for record_item in group
        ]
        flags = _pareto_flags_2d(
            [record_item.token_throughput_per_chip for record_item in group], ys
        )
        marked = [
            replace(record_item, is_pareto=flag)
            for record_item, flag in zip(group, flags, strict=True)
        ]
        marked.sort(key=lambda record_item: _pool_record_sort_key(record_item, sort_by))
        ordered.extend(marked)
    return tuple(ordered)


_ATTN_CORE_PARTITION_AXES = 4


def _count_core_valid_layouts(cores: int, *, has_moe: bool) -> int:
    """Count core-partitioned (PP, TP, DP, PCP, DCP, EP) tuples."""
    divisors = _divisors(cores)
    total = 0
    for pp_deg, tp_deg, dp_deg, pcp_deg in itertools.product(
        divisors, repeat=_ATTN_CORE_PARTITION_AXES
    ):
        if pp_deg * tp_deg * dp_deg * pcp_deg == cores:
            ep_mult = len(_divisors(tp_deg * dp_deg * pcp_deg)) if has_moe else 1
            total += len(_divisors(tp_deg)) * ep_mult
    return total


def _iter_disagg_ratio_pairs(
    eval_context: _SolveContext,
    prefill_topology: SliceTopology,
    decode_topology: SliceTopology,
    *,
    max_xp_override: int | None = None,
    max_yd_override: int | None = None,
) -> list[tuple[int, int, int]]:
    """Return valid (prefill_instances, decode_instances, total_chips) tuples."""
    if eval_context.pd_ratio is not None:
        xp_pin, yd_pin = eval_context.pd_ratio
        total_chips = (
            xp_pin * prefill_topology.num_chips + yd_pin * decode_topology.num_chips
        )
        if eval_context.max_chips is not None and total_chips > eval_context.max_chips:
            return []
        if (
            eval_context.max_instances is not None
            and (xp_pin + yd_pin) > eval_context.max_instances
        ):
            return []
        return [(xp_pin, yd_pin, total_chips)]
    effective_max = (
        eval_context.max_instances
        if eval_context.max_instances is not None
        else DEFAULT_MAX_INSTANCES
    )
    if effective_max < MIN_DISAGG_INSTANCES:
        return []
    max_single = effective_max - 1
    if eval_context.max_chips is not None:
        max_p = min(max_single, eval_context.max_chips // prefill_topology.num_chips)
        max_d = min(max_single, eval_context.max_chips // decode_topology.num_chips)
    else:
        max_p = max_single
        max_d = max_single
    if max_xp_override is not None:
        max_p = min(max_p, max_xp_override)
    if max_yd_override is not None:
        max_d = min(max_d, max_yd_override)
    pairs: list[tuple[int, int, int]] = []
    for prefill_instances in range(1, max_p + 1):
        for decode_instances in range(
            1, min(max_d, effective_max - prefill_instances) + 1
        ):
            total_chips = (
                prefill_instances * prefill_topology.num_chips
                + decode_instances * decode_topology.num_chips
            )
            if (
                eval_context.max_chips is not None
                and total_chips > eval_context.max_chips
            ):
                continue
            pairs.append((prefill_instances, decode_instances, total_chips))
    return pairs


def _count_disagg_ratio_slots(
    eval_context: _SolveContext,
    prefill_topology: SliceTopology,
    decode_topology: SliceTopology,
) -> int:
    """Count valid (prefill, decode) instance ratio slots for one slice pair."""
    return len(
        _iter_disagg_ratio_pairs(eval_context, prefill_topology, decode_topology)
    )


def _count_core_grid(
    eval_context: _SolveContext,
    topologies: Sequence[SliceTopology],
    *,
    include_pairs: bool,
) -> int:
    """Count core-partitioned (layout, concurrency) tuples across searched pools."""
    has_moe = eval_context.geometry.num_experts > 0
    conc_count = len(eval_context.scenario.concurrencies)
    total = 0
    if eval_context.mode in ("auto", "mixed") and (
        eval_context.max_instances is None or eval_context.max_instances >= 1
    ):
        for topology in topologies:
            if (
                eval_context.decode_topology is not None
                and topology.name != eval_context.decode_topology
            ):
                continue
            if (
                eval_context.max_chips is not None
                and topology.num_chips > eval_context.max_chips
            ):
                continue
            total += (
                _count_core_valid_layouts(topology.num_cores, has_moe=has_moe)
                * conc_count
            )
    if eval_context.mode in ("auto", "disagg") and include_pairs:
        for d_topo in topologies:
            if (
                eval_context.decode_topology is not None
                and d_topo.name != eval_context.decode_topology
            ):
                continue
            d_cnt = _count_core_valid_layouts(d_topo.num_cores, has_moe=has_moe)
            for p_topo in topologies:
                if (
                    eval_context.prefill_topology is not None
                    and p_topo.name != eval_context.prefill_topology
                ):
                    continue
                p_cnt = _count_core_valid_layouts(p_topo.num_cores, has_moe=has_moe)
                slots = _count_disagg_ratio_slots(eval_context, p_topo, d_topo)
                total += p_cnt * d_cnt * slots * conc_count
    return total


def _floor_times(floor_result: StepFloor) -> FloorTimes:
    """Convert StepFloor resource durations (s) into millisecond FloorTimes."""
    return FloorTimes(
        hbm_ms=floor_result.hbm_s * UNIT_MILLIS_PER_SECOND,
        matmul_ms=floor_result.matmul_s * UNIT_MILLIS_PER_SECOND,
        ici_ms=floor_result.ici_s * UNIT_MILLIS_PER_SECOND,
        chain_ms=floor_result.chain_s * UNIT_MILLIS_PER_SECOND,
        hop_latency_ms=floor_result.latency_assumed_s * UNIT_MILLIS_PER_SECOND,
        hard_floor_ms=floor_result.hard_floor_s * UNIT_MILLIS_PER_SECOND,
        binding=floor_result.binding,
        hbm_bound_ms=floor_result.hbm_bound_s * UNIT_MILLIS_PER_SECOND,
        matmul_bound_ms=floor_result.matmul_bound_s * UNIT_MILLIS_PER_SECOND,
        vector_bound_ms=floor_result.vector_bound_s * UNIT_MILLIS_PER_SECOND,
        coll_bound_ms=floor_result.coll_bound_s * UNIT_MILLIS_PER_SECOND,
    )


def _merge_floor_times_min(
    preferred: FloorTimes | None,
    other: FloorTimes | None,
) -> FloorTimes | None:
    """Return per-field minimum strict bounds across HBM-fitting residual modes."""
    if preferred is None:
        return other
    if other is None:
        return preferred
    return FloorTimes(
        hbm_ms=preferred.hbm_ms,
        matmul_ms=preferred.matmul_ms,
        ici_ms=preferred.ici_ms,
        chain_ms=preferred.chain_ms,
        hop_latency_ms=preferred.hop_latency_ms,
        hard_floor_ms=min(preferred.hard_floor_ms, other.hard_floor_ms),
        binding=preferred.binding,
        hbm_bound_ms=min(preferred.hbm_bound_ms, other.hbm_bound_ms),
        matmul_bound_ms=min(preferred.matmul_bound_ms, other.matmul_bound_ms),
        vector_bound_ms=min(preferred.vector_bound_ms, other.vector_bound_ms),
        coll_bound_ms=min(preferred.coll_bound_ms, other.coll_bound_ms),
    )


@dataclass
class _SolveContext:
    """Internal solver context capturing model, hardware, and scenario constants."""

    geometry: ModelGeometry
    dtypes: Dtypes
    chip: ChipSpec
    scenario: TrafficScenario
    mode: str
    max_chips: int | None
    max_instances: int | None
    hbm_reserve_gb: float
    hbm_limit_b: float
    hbm_limit_gb: float
    effective_accept_len: float
    spec_k: int
    ctx_full: int
    ctx_mid: int
    eff_isl: int
    n_chunks: int
    c_max: int
    shared_prefix_tokens: int = 0
    collective_latency_s: float = 0.0
    sort_by: SortBy = "throughput"
    prefill_topology: str | None = None
    decode_topology: str | None = None
    pd_ratio: tuple[int, int] | None = None
    activation_model: ExpertActivationModel = field(
        default_factory=ExpertActivationModel
    )
    enumerated: int = 0
    hbm_fit: int = 0
    decode_cache: dict[tuple[object, ...], StepFloor] = field(
        default_factory=dict, compare=False, hash=False
    )
    prefill_cache: dict[tuple[object, ...], StepFloor] = field(
        default_factory=dict, compare=False, hash=False
    )
    step_cache: dict[tuple[object, ...], StepFloor] = field(
        default_factory=dict, compare=False, hash=False
    )
    clamp_cache: dict[tuple[Parallel, int], tuple[int, HbmBreakdown] | None] = field(
        default_factory=dict, compare=False, hash=False
    )
    kv_fit_cache: dict[tuple[Parallel, int, int, float], tuple[float, float]] = field(
        default_factory=dict, compare=False, hash=False
    )
    prefill_unit_cap_cache: dict[tuple[str, Parallel], float] = field(
        default_factory=dict, compare=False, hash=False
    )
    decode_unit_cap_cache: dict[tuple[str, Parallel, int], float] = field(
        default_factory=dict, compare=False, hash=False
    )
    bound_decode_cache: dict[tuple[str, Parallel, int], StepFloor] = field(
        default_factory=dict, compare=False, hash=False
    )
    pool_plan_cache: dict[tuple[str, int, str, Parallel, int | None], PoolMeshPlan] = (
        field(default_factory=dict, compare=False, hash=False)
    )


@dataclass(frozen=True)
class _PrefillOption:
    """Feasible prefill configuration for disaggregated serving."""

    topology: SliceTopology
    parallel: Parallel
    ttft_ms: float
    floor_ttft_ms: float
    envelope_ttft_ms: float
    rate_p: float
    hbm_bd: HbmBreakdown
    binding: str
    w_base_b: float = 0.0
    kv_bytes_per_tok: float = 0.0
    fixed_kv_per_seq_b: float = 0.0
    offload_b_per_core: float = 0.0
    alt_sp: _PrefillOption | None = None


@dataclass(frozen=True)
class _DisaggDecodeState:
    """Decode configuration and memory bounds for disaggregated serving."""

    topology: SliceTopology
    parallel: Parallel
    w_base_b: float
    b_hbm_max: int
    kv_bytes_per_tok: float
    physical_batch_cap: int
    offload_b_per_core: float
    fixed_kv_per_seq_b: float = 0.0
    alt_sp: _DisaggDecodeState | None = None


@dataclass(frozen=True)
class _MixedPointBounds:
    """Precomputed base weights, capacity limits, and TTFT floor for mixed search."""

    w_base_b: float
    kv_bytes_per_tok: float
    b_hbm_max: int
    physical_batch_cap: int
    sf_ttft: StepFloor
    fixed_kv_per_seq_b: float = 0.0


@dataclass(frozen=True)
class _DisaggRatioTarget:
    """Concurrency and instance counts (xP, yD) for a disaggregated point."""

    concurrency: int
    prefill_instances: int
    decode_instances: int
    total_chips: int


@dataclass(frozen=True)
class _DisaggBatchStats:
    """Resolved decode batch size and effective prefill rate for a ratio target."""

    resident_batch: int
    local_batch: int
    ttft_eff_ms: float
    rate_p_eff: float


@dataclass(frozen=True)
class _PoolPlanInput:
    """Inputs needed to materialize one PoolMeshPlan instance pool."""

    role: Literal["mixed", "prefill", "decode"]
    instances: int
    topology: SliceTopology
    parallel: Parallel
    hbm_bd: HbmBreakdown
    hbm_limit_gb: float
    resident_batch_cap: int | None


def _build_pool_mesh_plan(spec: _PoolPlanInput) -> PoolMeshPlan:
    """Construct a PoolMeshPlan from topology, Parallel, and HBM usage."""
    axes = (
        spec.parallel.mesh_axes
        if spec.parallel.mesh_axes is not None
        else submesh_for_group(
            spec.parallel.tp, spec.topology, spec.topology.cores_per_chip
        )
    )
    return PoolMeshPlan(
        role=spec.role,
        instances=spec.instances,
        topology=spec.topology.name,
        chips_per_instance=spec.topology.num_chips,
        tp=spec.parallel.tp,
        dp=spec.parallel.dp,
        ep=spec.parallel.ep,
        pp=spec.parallel.pp,
        pcp=spec.parallel.pcp,
        kv_split=spec.parallel.kv_split,
        mesh_axes=axes,
        hbm_used_gb=spec.hbm_bd.total / UNIT_GIGA,
        hbm_limit_gb=spec.hbm_limit_gb,
        moe_dispatch=spec.parallel.moe_dispatch,
        hbm_weights_gb=spec.hbm_bd.weights / UNIT_GIGA,
        hbm_kv_gb=spec.hbm_bd.kv / UNIT_GIGA,
        hbm_state_gb=spec.hbm_bd.state / UNIT_GIGA,
        resident_batch_cap=spec.resident_batch_cap,
        residual=spec.parallel.residual,
    )


def _is_strictly_better_point(
    candidate: ParetoPoint,
    incumbent: ParetoPoint,
    sort_by: SortBy,
) -> bool:
    """Return True if candidate strictly beats incumbent on verdict + primary metric."""
    cand_rank = SLO_VERDICT_RANK[effective_slo_verdict(candidate)]
    inc_rank = SLO_VERDICT_RANK[effective_slo_verdict(incumbent)]
    if cand_rank != inc_rank:
        return cand_rank > inc_rank
    cand_pri, cand_sec = (
        (candidate.est_interactivity_tok_s, candidate.token_throughput_per_chip)
        if sort_by == "interactivity"
        else (candidate.token_throughput_per_chip, candidate.est_interactivity_tok_s)
    )
    inc_pri, inc_sec = (
        (incumbent.est_interactivity_tok_s, incumbent.token_throughput_per_chip)
        if sort_by == "interactivity"
        else (incumbent.token_throughput_per_chip, incumbent.est_interactivity_tok_s)
    )
    if abs(cand_pri - inc_pri) > 1e-12 * max(1.0, abs(inc_pri)):
        return cand_pri > inc_pri
    return cand_sec > inc_sec + 1e-12 * max(1.0, abs(inc_sec))


def _merge_collapsed_point_bounds(
    chosen: ParetoPoint,
    other: ParetoPoint,
    scenario: TrafficScenario,
    *,
    use_envelope: bool,
) -> ParetoPoint:
    """Merge per-field strict bounds across HBM-fitting residual modes."""
    merged_bound_tpot = min(chosen.bound_tpot_ms, other.bound_tpot_ms)
    merged_bound_ttft = min(chosen.bound_ttft_ms, other.bound_ttft_ms)
    merged_env_tpot = min(chosen.envelope_tpot_ms, other.envelope_tpot_ms)
    merged_env_ttft = min(chosen.envelope_ttft_ms, other.envelope_ttft_ms)
    merged_floor_tput = max(
        chosen.floor_throughput_per_chip, other.floor_throughput_per_chip
    )
    merged_min_seqs = min(chosen.min_decode_seqs, other.min_decode_seqs)
    merged_env_seqs = min(chosen.envelope_decode_seqs, other.envelope_decode_seqs)
    merged_verdict = _resolve_slo_verdict(
        scenario,
        chosen.est_ttft_ms,
        chosen.est_tpot_ms,
        merged_env_ttft if use_envelope else merged_bound_ttft,
        merged_env_tpot if use_envelope else merged_bound_tpot,
    )
    if (
        effective_slo_verdict(chosen) == SLO_VERDICT_RULED_OUT
        and effective_slo_verdict(other) == SLO_VERDICT_RULED_OUT
    ):
        merged_verdict = SLO_VERDICT_RULED_OUT
    return replace(
        chosen,
        slo_met=merged_verdict == SLO_VERDICT_ESTIMATE_MEETS,
        slo_verdict=merged_verdict,
        bound_tpot_ms=merged_bound_tpot,
        bound_ttft_ms=merged_bound_ttft,
        envelope_tpot_ms=merged_env_tpot,
        envelope_ttft_ms=merged_env_ttft,
        floor_throughput_per_chip=merged_floor_tput,
        min_decode_seqs=merged_min_seqs,
        envelope_decode_seqs=merged_env_seqs,
        decode_floor=_merge_floor_times_min(chosen.decode_floor, other.decode_floor),
    )


def _is_strictly_better_decode_record(
    candidate: PoolFloorRecord,
    incumbent: PoolFloorRecord,
    sort_by: SortBy,
) -> bool:
    """Return True iff candidate decode record strictly beats incumbent."""
    if candidate.slo_met != incumbent.slo_met:
        return candidate.slo_met
    cand_speed = candidate.interactivity_tok_s or 0.0
    inc_speed = incumbent.interactivity_tok_s or 0.0
    cand_pri, cand_sec = (
        (cand_speed, candidate.token_throughput_per_chip)
        if sort_by == "interactivity"
        else (candidate.token_throughput_per_chip, cand_speed)
    )
    inc_pri, inc_sec = (
        (inc_speed, incumbent.token_throughput_per_chip)
        if sort_by == "interactivity"
        else (incumbent.token_throughput_per_chip, inc_speed)
    )
    if abs(cand_pri - inc_pri) > 1e-12 * max(1.0, abs(inc_pri)):
        return cand_pri > inc_pri
    return cand_sec > inc_sec + 1e-12 * max(1.0, abs(inc_sec))


_FIXED_POINT_MAX_ITERS: Final[int] = 12
_FIXED_POINT_TOL: Final[float] = 1e-6
_RATE_FLOOR_EPS: Final[float] = 1e-9
_CAP_DENOM_EPS: Final[float] = 1e-12


def _within_slo_bounds(
    scenario: TrafficScenario,
    ttft_ms: float,
    tpot_ms: float,
) -> bool:
    """Return True when both TTFT and TPOT stay within any configured SLOs."""
    ttft_ok = scenario.ttft_slo_ms is None or ttft_ms <= scenario.ttft_slo_ms
    tpot_ok = scenario.tpot_slo_ms is None or tpot_ms <= scenario.tpot_slo_ms
    return ttft_ok and tpot_ok


def _resolve_slo_verdict(
    scenario: TrafficScenario,
    est_ttft_ms: float,
    est_tpot_ms: float,
    bound_ttft_ms: float,
    bound_tpot_ms: float,
) -> SloVerdict:
    """Classify lower-bound vs modeled-estimate SLO feasibility verdict."""
    if not _within_slo_bounds(scenario, bound_ttft_ms, bound_tpot_ms):
        return SLO_VERDICT_RULED_OUT
    return (
        SLO_VERDICT_ESTIMATE_MEETS
        if _within_slo_bounds(scenario, est_ttft_ms, est_tpot_ms)
        else SLO_VERDICT_UNDECIDED
    )


def _apply_capacity_ceiling_verdict(
    scenario: TrafficScenario,
    verdict: SloVerdict,
    *,
    concurrency: int = 0,
    pool_capacity_req_s: float | None = None,
) -> SloVerdict:
    """Rule out closed-loop dual-SLO workloads exceeding pool capacity ceiling."""
    if (
        verdict == SLO_VERDICT_RULED_OUT
        or pool_capacity_req_s is None
        or concurrency <= 0
        or scenario.think_time_s is None
        or scenario.ttft_slo_ms is None
        or scenario.tpot_slo_ms is None
    ):
        return verdict
    x_min = float(concurrency) / max(
        _RATE_FLOOR_EPS,
        scenario.ttft_slo_ms / UNIT_MILLIS_PER_SECOND
        + float(_decode_tokens(scenario.osl))
        * (scenario.tpot_slo_ms / UNIT_MILLIS_PER_SECOND)
        + scenario.think_time_s,
    )
    return SLO_VERDICT_RULED_OUT if pool_capacity_req_s < x_min else verdict


class _BoundOpts(TypedDict, total=False):
    """Keyword options for analytical physical bound entry points."""

    prefill_par: Parallel | None
    prefill_topo: SliceTopology | None
    decode_par: Parallel | None
    decode_topo: SliceTopology | None
    replica_batch_cap: int
    concurrency: int
    prefill_instances: int
    decode_instances: int
    spec_k: int | None
    activation_model: ExpertActivationModel | None
    total_chips: int | None


@dataclass(frozen=True, slots=True)
class _BoundDecodeInput:
    """Pure physical parameters for evaluating decode step/spec floors."""

    geometry: ModelGeometry
    dtypes: Dtypes
    chip: ChipSpec
    scenario: TrafficScenario
    decode_par: Parallel
    decode_topo: SliceTopology
    spec_k: int
    activation_model: ExpertActivationModel
    ctx_mid: int
    effective_accept_len: float
    shared_prefix_tokens: int = 0


def _shared_prefix_tokens(scenario: TrafficScenario) -> int:
    """Return rounded shared-prefix token count for one scenario."""
    return round(scenario.prefix_hit_rate * scenario.isl)


def _resolve_spec_k(
    geometry: ModelGeometry,
    scenario: TrafficScenario,
    spec_k: int | None = None,
) -> int:
    """Resolve effective speculative draft step count without touching estimates."""
    if spec_k is not None:
        return spec_k
    if scenario.spec_tokens is not None:
        return scenario.spec_tokens
    effective_accept_len = max(1.0, scenario.accept_len)
    if effective_accept_len > 1.0:
        return max(geometry.mtp_layers, math.ceil(effective_accept_len) - 1)
    return 0


def _steady_state_batch_cap_phys(
    eval_context: _SolveContext,
    parallel: Parallel,
) -> int:
    """Compute unrounded steady-state physical HBM batch cap with shared prefix."""
    geometry = eval_context.geometry
    dtypes = eval_context.dtypes
    chip = eval_context.chip
    scenario = eval_context.scenario
    w_base_phys = hbm_breakdown_per_core_bytes(
        geometry, dtypes, parallel, SeqKvLayout(0, 0), chip=chip
    ).weights
    state_phys = hbm_breakdown_per_core_bytes(
        geometry, dtypes, parallel, SeqKvLayout(1, 0), chip=chip
    ).state
    h_rate = scenario.prefix_hit_rate
    shared_toks = h_rate * float(scenario.isl)
    avg_priv_toks = (1.0 - h_rate) * float(scenario.isl) + (
        float(_decode_tokens(scenario.osl)) / 2.0
    )
    full_kv_per_tok, _ = kv_bytes_per_core_parts(
        geometry, dtypes, parallel, 0, physical=False, chip=chip
    )
    if geometry.sliding_window > 0:
        _, swa_per_tok = kv_bytes_per_core_parts(
            geometry, dtypes, parallel, 1, physical=False, chip=chip
        )
        swa_win = float(geometry.sliding_window)
        priv_swa = min(avg_priv_toks, swa_win)
        shared_swa = max(0.0, min(shared_toks, swa_win - priv_swa))
        shared_kv = shared_toks * full_kv_per_tok + shared_swa * swa_per_tok
        avg_priv_kv = avg_priv_toks * full_kv_per_tok + priv_swa * swa_per_tok
    else:
        shared_kv = shared_toks * full_kv_per_tok
        avg_priv_kv = avg_priv_toks * full_kv_per_tok
    avail = eval_context.hbm_limit_b - w_base_phys - shared_kv
    if avail < 0.0:
        return 0
    denom = state_phys + avg_priv_kv
    if denom <= 0.0:
        return max(1, eval_context.c_max)
    return math.floor(avail / denom)


def _eval_bound_decode_floor(bound_in: _BoundDecodeInput, batch: int) -> StepFloor:
    """Evaluate a pure physical decode step floor without latency calibration."""
    opts = FloorOptions(
        dtypes=bound_in.dtypes,
        topology=bound_in.decode_topo,
        activation_model=bound_in.activation_model,
    )
    if bound_in.spec_k > 0:
        return spec_cycle_floor(
            bound_in.geometry,
            bound_in.chip,
            bound_in.decode_par,
            SpecShape(
                batch=batch,
                context=bound_in.ctx_mid,
                spec_k=bound_in.spec_k,
                shared_prefix_tokens=bound_in.shared_prefix_tokens,
            ),
            opts,
        )
    return step_floor(
        bound_in.geometry,
        bound_in.chip,
        bound_in.decode_par,
        StepShape(
            batch=batch,
            context=bound_in.ctx_mid,
            shared_prefix_tokens=bound_in.shared_prefix_tokens,
        ),
        opts,
    )


def _eval_pp_bound_decode_cycle(
    bound_in: _BoundDecodeInput,
    batch: int,
) -> StepFloor:
    """Evaluate PP-aware decode step floor: max(T_full(max(1, B//pp)), T_full(B)/pp)."""
    eff_batch = max(1, batch)
    pp = bound_in.decode_par.pp
    if pp <= 1:
        return _eval_bound_decode_floor(bound_in, eff_batch)
    bound_pp1 = replace(bound_in, decode_par=replace(bound_in.decode_par, pp=1))
    floor_mb = max(1, eff_batch // pp)
    sf_mb = _eval_bound_decode_floor(bound_pp1, floor_mb)
    sf_full = (
        sf_mb
        if floor_mb == eff_batch
        else _eval_bound_decode_floor(bound_pp1, eff_batch)
    )
    pp_f = float(pp)
    hbm_b = max(sf_mb.hbm_bound_s, sf_full.hbm_bound_s / pp_f)
    mm_b = max(sf_mb.matmul_bound_s, sf_full.matmul_bound_s / pp_f)
    vec_b = max(sf_mb.vector_bound_s, sf_full.vector_bound_s / pp_f)
    coll_b = max(sf_mb.coll_bound_s, sf_full.coll_bound_s / pp_f)
    hard_s = max(
        sf_mb.hard_floor_s, sf_full.hard_floor_s / pp_f, hbm_b, mm_b, vec_b, coll_b
    )
    env_s = max(sf_mb.envelope_floor_s, sf_full.envelope_floor_s / pp_f)
    return replace(
        sf_mb,
        hard_floor_s=hard_s,
        envelope_floor_s=env_s,
        hbm_bound_s=hbm_b,
        matmul_bound_s=mm_b,
        vector_bound_s=vec_b,
        coll_bound_s=coll_b,
    )


def _eval_bound_prefill_floor(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    opts: _BoundOpts,
) -> StepFloor:
    """Evaluate a pure physical prefill step floor at scenario prefix_hit_rate."""
    pref_par = opts.get("prefill_par")
    pref_topo = opts.get("prefill_topo")
    if pref_par is None or pref_topo is None:
        msg = "prefill_par and prefill_topo are required for TTFT bounds"
        raise ValueError(msg)
    act_model = opts.get("activation_model") or ExpertActivationModel()
    return step_floor(
        geometry,
        chip,
        pref_par,
        StepShape(
            batch=0,
            rows_per_seq=0,
            context=0,
            state_writes=0,
            prefill_tokens=scenario.isl,
            prefix_hit_rate=scenario.prefix_hit_rate,
        ),
        FloorOptions(
            dtypes=dtypes,
            topology=pref_topo,
            activation_model=act_model,
        ),
    )


def bound_ttft_ms(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_BoundOpts],
) -> float:
    """Return physical TTFT lower bound (ms) strictly at scenario prefix_hit_rate."""
    floor_result = _eval_bound_prefill_floor(geometry, dtypes, chip, scenario, opts)
    return floor_result.hard_floor_s * UNIT_MILLIS_PER_SECOND


def envelope_ttft_ms(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_BoundOpts],
) -> float:
    """Return activation-envelope TTFT bound (ms) at scenario prefix_hit_rate."""
    floor_result = _eval_bound_prefill_floor(geometry, dtypes, chip, scenario, opts)
    return floor_result.envelope_floor_s * UNIT_MILLIS_PER_SECOND


def _eval_unpadded_prefill_buckets(
    bound_in: _BoundDecodeInput,
    pref_par: Parallel,
    pref_topo: SliceTopology,
    *,
    zero_weights: bool,
) -> tuple[float, float, float, LinkCutLoad]:
    """Return unpadded prefill (hbm, matmul, vector, LinkCutLoad) buckets."""
    scenario = bound_in.scenario
    if scenario.isl <= 0:
        return (0.0, 0.0, 0.0, LinkCutLoad())
    eff_isl_floor = max(
        max(1, pref_par.pcp),
        math.floor((1.0 - scenario.prefix_hit_rate) * float(scenario.isl)),
    )
    eff_hit_rate = (
        max(0.0, float(scenario.isl - eff_isl_floor) / float(scenario.isl) - 1e-12)
        if eff_isl_floor < scenario.isl
        else 0.0
    )
    eval_par = replace(pref_par, pp=1) if pref_par.pp > 1 else pref_par
    hbm_s, mm_s, vec_s, load = unpadded_strict_resource_buckets(
        bound_in.geometry,
        bound_in.chip,
        eval_par,
        StepShape(
            batch=0,
            rows_per_seq=1,
            context=0,
            prefill_tokens=scenario.isl,
            prefix_hit_rate=eff_hit_rate,
        ),
        UnpaddedFloorOptions(
            dtypes=bound_in.dtypes,
            topology=pref_topo,
            zero_weights=zero_weights,
        ),
    )
    if pref_par.pp > 1:
        inv_pp = 1.0 / float(pref_par.pp)
        return (hbm_s * inv_pp, mm_s * inv_pp, vec_s * inv_pp, load.scaled(inv_pp))
    return (hbm_s, mm_s, vec_s, load)


def _eval_unpadded_decode_buckets(
    bound_in: _BoundDecodeInput,
    replica_batch_cap: int,
) -> tuple[float, float, float, LinkCutLoad]:
    """Return PP-aware unpadded decode (hbm, matmul, vector, LinkCutLoad) buckets."""
    eff_batch = max(1, replica_batch_cap)
    pp = bound_in.decode_par.pp
    if pp <= 1:
        return unpadded_strict_resource_buckets(
            bound_in.geometry,
            bound_in.chip,
            bound_in.decode_par,
            StepShape(
                batch=eff_batch,
                rows_per_seq=bound_in.spec_k + 1,
                context=bound_in.ctx_mid,
                state_writes=1,
                shared_prefix_tokens=bound_in.shared_prefix_tokens,
            ),
            UnpaddedFloorOptions(
                dtypes=bound_in.dtypes,
                topology=bound_in.decode_topo,
                spec_k=bound_in.spec_k,
                zero_weights=False,
            ),
        )
    par_pp1 = replace(bound_in.decode_par, pp=1)
    floor_mb = max(1, eff_batch // pp)
    opts = UnpaddedFloorOptions(
        dtypes=bound_in.dtypes,
        topology=bound_in.decode_topo,
        spec_k=bound_in.spec_k,
        zero_weights=False,
    )
    hbm_mb, mm_mb, vec_mb, load_mb = unpadded_strict_resource_buckets(
        bound_in.geometry,
        bound_in.chip,
        par_pp1,
        StepShape(
            batch=floor_mb,
            rows_per_seq=bound_in.spec_k + 1,
            context=bound_in.ctx_mid,
            state_writes=1,
            shared_prefix_tokens=bound_in.shared_prefix_tokens,
        ),
        opts,
    )
    if floor_mb == eff_batch:
        return (hbm_mb, mm_mb, vec_mb, load_mb)
    hbm_f, mm_f, vec_f, load_f = unpadded_strict_resource_buckets(
        bound_in.geometry,
        bound_in.chip,
        par_pp1,
        StepShape(
            batch=eff_batch,
            rows_per_seq=bound_in.spec_k + 1,
            context=bound_in.ctx_mid,
            state_writes=1,
            shared_prefix_tokens=bound_in.shared_prefix_tokens,
        ),
        opts,
    )
    inv_pp = 1.0 / float(pp)
    scaled_load_f = load_f.scaled(inv_pp)
    coll_mb = eval_link_subset_bound_s(load_mb, bound_in.chip, bound_in.decode_topo)
    coll_f = eval_link_subset_bound_s(
        scaled_load_f, bound_in.chip, bound_in.decode_topo
    )
    return (
        max(hbm_mb, hbm_f * inv_pp),
        max(mm_mb, mm_f * inv_pp),
        max(vec_mb, vec_f * inv_pp),
        scaled_load_f if coll_f > coll_mb else load_mb,
    )


def _mixed_capacity_work_bound(
    bound_in: _BoundDecodeInput,
    replica_batch_cap: int,
) -> float:
    """Return mixed-pool capacity ceiling (req/s) with link-axis cuts."""
    dec_steps = _decode_steps_per_req(
        bound_in.scenario.osl, bound_in.effective_accept_len
    )
    pre_hbm, pre_mm, pre_vec, pre_load = _eval_unpadded_prefill_buckets(
        bound_in,
        bound_in.decode_par,
        bound_in.decode_topo,
        zero_weights=True,
    )
    if dec_steps <= 0.0:
        pre_coll = eval_link_subset_bound_s(
            pre_load, bound_in.chip, bound_in.decode_topo
        )
        req_time_s = max(pre_hbm, pre_mm, pre_vec, pre_coll)
        if req_time_s <= 0.0:
            return math.inf
        return float(max(1, bound_in.decode_par.dp)) / max(_CAP_DENOM_EPS, req_time_s)
    dec_hbm, dec_mm, dec_vec, dec_load = _eval_unpadded_decode_buckets(
        bound_in, replica_batch_cap
    )
    dec_scale = dec_steps / float(max(1, replica_batch_cap))
    comb_load = pre_load.add(dec_load.scaled(dec_scale))
    coll_req_s = eval_link_subset_bound_s(
        comb_load, bound_in.chip, bound_in.decode_topo
    )
    req_time_s = max(
        pre_hbm + dec_scale * dec_hbm,
        pre_mm + dec_scale * dec_mm,
        pre_vec + dec_scale * dec_vec,
        coll_req_s,
    )
    return float(max(1, bound_in.decode_par.dp)) / max(_CAP_DENOM_EPS, req_time_s)


def _disagg_capacity_work_bound(
    bound_in: _BoundDecodeInput,
    pref_par: Parallel,
    pref_topo: SliceTopology,
    ratio_cap: tuple[int, int, int],
) -> float:
    """Return disagg-pool capacity ceiling (req/s) across P and D pools."""
    prefill_instances, decode_instances, replica_batch_cap = ratio_cap
    dec_steps = _decode_steps_per_req(
        bound_in.scenario.osl, bound_in.effective_accept_len
    )
    pre_hbm, pre_mm, pre_vec, pre_load = _eval_unpadded_prefill_buckets(
        bound_in, pref_par, pref_topo, zero_weights=True
    )
    pre_coll = eval_link_subset_bound_s(pre_load, bound_in.chip, pref_topo)
    pre_time = max(pre_hbm, pre_mm, pre_vec, pre_coll)
    prefill_capacity_rate = (
        float(max(1, prefill_instances * pref_par.dp)) / max(_CAP_DENOM_EPS, pre_time)
        if pre_time > 0.0
        else math.inf
    )
    if dec_steps <= 0.0:
        return prefill_capacity_rate
    dec_hbm, dec_mm, dec_vec, dec_load = _eval_unpadded_decode_buckets(
        bound_in, replica_batch_cap
    )
    dec_coll = eval_link_subset_bound_s(dec_load, bound_in.chip, bound_in.decode_topo)
    dec_scale = dec_steps / float(max(1, replica_batch_cap))
    dec_time = dec_scale * max(dec_hbm, dec_mm, dec_vec, dec_coll)
    decode_capacity_rate = float(
        max(1, decode_instances * bound_in.decode_par.dp)
    ) / max(_CAP_DENOM_EPS, dec_time)
    return min(prefill_capacity_rate, decode_capacity_rate)


def _context_to_bound_input(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
) -> _BoundDecodeInput:
    """Extract pure physical bound input from a solver evaluation context."""
    return _BoundDecodeInput(
        geometry=eval_context.geometry,
        dtypes=eval_context.dtypes,
        chip=eval_context.chip,
        scenario=eval_context.scenario,
        decode_par=parallel,
        decode_topo=topology,
        spec_k=eval_context.spec_k,
        activation_model=eval_context.activation_model,
        ctx_mid=eval_context.ctx_mid,
        effective_accept_len=eval_context.effective_accept_len,
        shared_prefix_tokens=eval_context.shared_prefix_tokens,
    )


def _solve_fixed_point_residency(
    bound_in: _BoundDecodeInput,
    concurrency: int,
    replica_batch_cap: int,
    idp_val: int,
    x_cap_eff: float,
) -> tuple[float, StepFloor]:
    """Run bottom-up fixed-point pure-decode hard_floor residency fixed-point loop."""
    scenario = bound_in.scenario
    ttft_slo_ms = scenario.ttft_slo_ms
    think = scenario.think_time_s
    dec_toks = float(_decode_tokens(scenario.osl))
    if ttft_slo_ms is None or think is None or dec_toks <= 0.0:
        return (1.0, _eval_pp_bound_decode_cycle(bound_in, 1))
    ttft_upper_s = ttft_slo_ms / UNIT_MILLIS_PER_SECOND
    max_cap_seqs = float(min(concurrency, replica_batch_cap * idp_val))
    b_dec = 1.0
    for _ in range(_FIXED_POINT_MAX_ITERS):
        b_rep = max(1, min(replica_batch_cap, math.floor(b_dec / float(idp_val))))
        decode_floor = _eval_pp_bound_decode_cycle(bound_in, b_rep)
        t_dec_s = dec_toks * (decode_floor.hard_floor_s / bound_in.effective_accept_len)
        x_rate = min(
            x_cap_eff,
            float(concurrency) / max(_RATE_FLOOR_EPS, ttft_upper_s + t_dec_s + think),
        )
        b_next = max(1.0, min(max_cap_seqs, x_rate * t_dec_s))
        if abs(b_next - b_dec) <= _FIXED_POINT_TOL:
            b_dec = b_next
            break
        b_dec = b_next
    floor_b_replica = max(1, min(replica_batch_cap, math.floor(b_dec / float(idp_val))))
    return (b_dec, _eval_pp_bound_decode_cycle(bound_in, floor_b_replica))


def _build_bound_decode_input(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    opts: _BoundOpts,
) -> tuple[_BoundDecodeInput, int]:
    """Validate and construct physical decode input + effective b_cap_single."""
    decode_par = opts.get("decode_par")
    decode_topo = opts.get("decode_topo")
    if decode_par is None or decode_topo is None:
        msg = "decode_par and decode_topo are required for physical decode bounds"
        raise ValueError(msg)
    replica_batch_cap = max(1, opts.get("replica_batch_cap", 1))
    spec_k = _resolve_spec_k(geometry, scenario, opts.get("spec_k"))
    act_model = opts.get("activation_model") or ExpertActivationModel()
    effective_accept_len = max(1.0, scenario.accept_len)
    ctx_mid = scenario.isl + scenario.osl // 2
    bound_in = _BoundDecodeInput(
        geometry=geometry,
        dtypes=dtypes,
        chip=chip,
        scenario=scenario,
        decode_par=decode_par,
        decode_topo=decode_topo,
        spec_k=spec_k,
        activation_model=act_model,
        ctx_mid=ctx_mid,
        effective_accept_len=effective_accept_len,
        shared_prefix_tokens=_shared_prefix_tokens(scenario),
    )
    return (bound_in, replica_batch_cap)


def _resolve_pool_capacity_context(
    bound_in: _BoundDecodeInput,
    replica_batch_cap: int,
    opts: _BoundOpts,
) -> tuple[float, Parallel, SliceTopology, int, int]:
    """Return pool capacity bound, prefill pool, and instance counts."""
    pref_par = opts.get("prefill_par")
    pref_topo = opts.get("prefill_topo")
    if pref_par is not None and pref_topo is not None:
        prefill_instances = max(1, opts.get("prefill_instances", 1))
        decode_instances = max(1, opts.get("decode_instances", 1))
        capacity_rate_bound = _disagg_capacity_work_bound(
            bound_in,
            pref_par,
            pref_topo,
            (prefill_instances, decode_instances, replica_batch_cap),
        )
        return (
            capacity_rate_bound,
            pref_par,
            pref_topo,
            prefill_instances,
            decode_instances,
        )
    return (
        _mixed_capacity_work_bound(bound_in, replica_batch_cap),
        bound_in.decode_par,
        bound_in.decode_topo,
        1,
        1,
    )


def pool_capacity_bound(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_BoundOpts],
) -> float:
    """Return isolated unpadded physical pool request capacity ceiling (req/s)."""
    bound_in, replica_batch_cap = _build_bound_decode_input(
        geometry, dtypes, chip, scenario, opts
    )
    return _resolve_pool_capacity_context(bound_in, replica_batch_cap, opts)[0]


def _resolve_residency_floor(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    opts: _BoundOpts,
) -> tuple[_BoundDecodeInput, StepFloor]:
    """Compute physical fixed-point decode step floor for mixed or disagg bounds."""
    bound_in, replica_batch_cap = _build_bound_decode_input(
        geometry, dtypes, chip, scenario, opts
    )
    concurrency = max(1, opts.get("concurrency", 1))
    capacity_rate_bound, ttft_par, ttft_topo, prefill_instances, decode_instances = (
        _resolve_pool_capacity_context(bound_in, replica_batch_cap, opts)
    )
    if opts.get("prefill_par") is not None and opts.get("prefill_topo") is not None:
        b_ttft_ms = bound_ttft_ms(
            geometry,
            dtypes,
            chip,
            scenario,
            prefill_par=ttft_par,
            prefill_topo=ttft_topo,
            activation_model=bound_in.activation_model,
        )
        rate_p_floor = float(max(1, prefill_instances * ttft_par.dp)) / max(
            _RATE_FLOOR_EPS, b_ttft_ms / UNIT_MILLIS_PER_SECOND
        )
        capacity_rate_bound = min(capacity_rate_bound, rate_p_floor)
    _, floor_result = _solve_fixed_point_residency(
        bound_in,
        concurrency,
        replica_batch_cap,
        max(1, decode_instances * bound_in.decode_par.dp),
        capacity_rate_bound,
    )
    return (bound_in, floor_result)


def bound_tpot_ms(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_BoundOpts],
) -> float:
    """Return physical TPOT lower bound (ms) under fixed-point residency."""
    bound_in, floor_result = _resolve_residency_floor(
        geometry, dtypes, chip, scenario, opts
    )
    return (
        floor_result.hard_floor_s * UNIT_MILLIS_PER_SECOND
    ) / bound_in.effective_accept_len


def envelope_tpot_ms(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_BoundOpts],
) -> float:
    """Return TTFT-SLO-conditional activation-envelope TPOT bound (ms).

    The fixed-point residency lower bound assumes TTFT stays within
    ``scenario.ttft_slo_ms``; modeled estimates that violate ``ttft_slo_ms``
    can have lower residency and therefore lower ``est_tpot_ms``.
    """
    bound_in, floor_result = _resolve_residency_floor(
        geometry, dtypes, chip, scenario, opts
    )
    return (
        floor_result.envelope_floor_s * UNIT_MILLIS_PER_SECOND
    ) / bound_in.effective_accept_len


def request_rate_bound(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_BoundOpts],
) -> float:
    """Return isolated physical request rate upper bound (req/s)."""
    bound_in, replica_batch_cap = _build_bound_decode_input(
        geometry, dtypes, chip, scenario, opts
    )
    concurrency = max(1, opts.get("concurrency", 1))
    capacity_rate_bound, ttft_par, ttft_topo, _, _ = _resolve_pool_capacity_context(
        bound_in, replica_batch_cap, opts
    )
    if scenario.think_time_s is None:
        return capacity_rate_bound
    b_ttft_s = (
        bound_ttft_ms(
            geometry,
            dtypes,
            chip,
            scenario,
            prefill_par=ttft_par,
            prefill_topo=ttft_topo,
            activation_model=bound_in.activation_model,
        )
        / UNIT_MILLIS_PER_SECOND
    )
    dec_toks = float(_decode_tokens(scenario.osl))
    dec_b1_s = (
        dec_toks
        * (
            _eval_pp_bound_decode_cycle(bound_in, 1).hard_floor_s
            / bound_in.effective_accept_len
        )
        if dec_toks > 0.0
        else 0.0
    )
    return min(
        capacity_rate_bound,
        float(concurrency)
        / max(_RATE_FLOOR_EPS, b_ttft_s + dec_b1_s + scenario.think_time_s),
    )


def floor_throughput_per_chip(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_BoundOpts],
) -> float:
    """Return isolated physical per-chip token throughput ceiling (tok/s/chip)."""
    x_rate_bound = request_rate_bound(geometry, dtypes, chip, scenario, **opts)
    total_chips = opts.get("total_chips")
    if total_chips is None:
        dec_topo = opts.get("decode_topo")
        pref_topo = opts.get("prefill_topo")
        if dec_topo is None:
            msg = "decode_topo or total_chips is required"
            raise ValueError(msg)
        if opts.get("prefill_par") is not None and pref_topo is not None:
            prefill_instances = max(1, opts.get("prefill_instances", 1))
            decode_instances = max(1, opts.get("decode_instances", 1))
            total_chips = (
                prefill_instances * pref_topo.num_chips
                + decode_instances * dec_topo.num_chips
            )
        else:
            total_chips = dec_topo.num_chips
    return (x_rate_bound * float(scenario.isl + scenario.osl)) / float(
        max(1, total_chips)
    )


def _eval_decode_floor(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    batch: int,
    shared_prefix_tokens: int | None = None,
) -> StepFloor:
    """Evaluate single-step or speculative-cycle floor at midpoint context."""
    shared_tok = (
        eval_context.shared_prefix_tokens
        if shared_prefix_tokens is None
        else shared_prefix_tokens
    )
    cache_key = (topology.name, parallel, batch, shared_tok)
    cached = eval_context.decode_cache.get(cache_key)
    if cached is not None:
        return cached
    opts = FloorOptions(
        dtypes=eval_context.dtypes,
        collective_latency_s=eval_context.collective_latency_s,
        topology=topology,
        activation_model=eval_context.activation_model,
    )
    if eval_context.spec_k > 0:
        result = spec_cycle_floor(
            eval_context.geometry,
            eval_context.chip,
            parallel,
            SpecShape(
                batch=batch,
                context=eval_context.ctx_mid,
                spec_k=eval_context.spec_k,
                shared_prefix_tokens=shared_tok,
            ),
            opts,
        )
    else:
        result = step_floor(
            eval_context.geometry,
            eval_context.chip,
            parallel,
            StepShape(
                batch=batch,
                context=eval_context.ctx_mid,
                shared_prefix_tokens=shared_tok,
            ),
            opts,
        )
    eval_context.decode_cache[cache_key] = result
    return result


def _eval_pp_decode_stages(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    batch: int,
    shared_prefix_tokens: int | None,
    *,
    need_hard_floor: bool,
) -> tuple[float, StepFloor, StepFloor | None, StepFloor | None]:
    """Compute PP decode serial cycle and optional pp=1 lower-bound floor stages."""
    pp = parallel.pp
    num_microbatches = min(pp, batch)
    micro_batch = math.ceil(batch / num_microbatches)
    stage_floor = _eval_decode_floor(
        eval_context, topology, parallel, micro_batch, shared_prefix_tokens
    )
    par_pp1 = replace(parallel, pp=1)
    if batch < pp:
        full_mb_floor = _eval_decode_floor(
            eval_context, topology, par_pp1, micro_batch, shared_prefix_tokens
        )
        serial_cycle_s = max(
            full_mb_floor.serial_estimate_s,
            float(num_microbatches) * stage_floor.serial_estimate_s,
        )
    else:
        serial_cycle_s = float(pp) * stage_floor.serial_estimate_s
    if not need_hard_floor:
        return serial_cycle_s, stage_floor, None, None
    floor_mb = max(1, batch // pp)
    full_floor_mb = _eval_decode_floor(
        eval_context, topology, par_pp1, floor_mb, shared_prefix_tokens
    )
    full_batch_floor = (
        full_floor_mb
        if floor_mb == batch
        else _eval_decode_floor(
            eval_context, topology, par_pp1, batch, shared_prefix_tokens
        )
    )
    return serial_cycle_s, stage_floor, full_floor_mb, full_batch_floor


def _eval_pp_decode_cycle(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    local_batch: int,
    shared_prefix_tokens: int | None = None,
    *,
    need_hard_floor: bool = False,
) -> tuple[float, float, str]:
    """Return (serial_cycle_s, hard_cycle_s, binding) accounting for PP stages."""
    batch = max(1, local_batch)
    pp = parallel.pp
    if pp <= 1:
        floor = _eval_decode_floor(
            eval_context, topology, parallel, batch, shared_prefix_tokens
        )
        return floor.serial_estimate_s, floor.hard_floor_s, floor.binding
    serial_cycle_s, stage_floor, full_floor_mb, full_batch_floor = (
        _eval_pp_decode_stages(
            eval_context,
            topology,
            parallel,
            batch,
            shared_prefix_tokens,
            need_hard_floor=need_hard_floor,
        )
    )
    if full_floor_mb is None or full_batch_floor is None:
        return serial_cycle_s, 0.0, stage_floor.binding
    hard_cycle_s = max(
        full_floor_mb.hard_floor_s,
        full_batch_floor.hard_floor_s / float(pp),
    )
    return serial_cycle_s, hard_cycle_s, stage_floor.binding


def _eval_cached_step_floor(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    shape: StepShape,
) -> StepFloor:
    """Evaluate step_floor backed by per-solve memoization."""
    cache_key = (topology.name, parallel, shape)
    cached = eval_context.step_cache.get(cache_key)
    if cached is not None:
        return cached
    result = step_floor(
        eval_context.geometry,
        eval_context.chip,
        parallel,
        shape,
        FloorOptions(
            dtypes=eval_context.dtypes,
            collective_latency_s=eval_context.collective_latency_s,
            topology=topology,
            activation_model=eval_context.activation_model,
        ),
    )
    eval_context.step_cache[cache_key] = result
    return result


def _eval_pp_step_serial(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    base_shape: StepShape,
    local_batch: int,
) -> tuple[float, str]:
    """Return PP-serialized (serial_cycle_s, binding) for a batched step shape."""
    batch = max(1, local_batch)
    pp = parallel.pp
    if pp <= 1:
        floor = _eval_cached_step_floor(
            eval_context, topology, parallel, replace(base_shape, batch=batch)
        )
        return floor.serial_estimate_s, floor.binding
    num_microbatches = min(pp, batch)
    micro_batch = math.ceil(batch / num_microbatches)
    if base_shape.prefill_tokens > 0 and num_microbatches > 1:
        cached_prior = float(base_shape.prefill_tokens) * base_shape.prefix_hit_rate
        total_new = float(base_shape.prefill_tokens) * max(
            0.0, 1.0 - base_shape.prefix_hit_rate
        )
        mb_new = total_new / float(num_microbatches)
        avg_prior_mb = cached_prior + CAUSAL_MIDPOINT_HALF * (total_new - mb_new)
        mb_span = avg_prior_mb + mb_new
        mb_shape = replace(
            base_shape,
            batch=micro_batch,
            prefill_tokens=max(1, round(mb_span)),
            prefix_hit_rate=(avg_prior_mb / mb_span if mb_span > 0.0 else 0.0),
        )
    else:
        mb_shape = replace(base_shape, batch=micro_batch)
    stage_floor = _eval_cached_step_floor(eval_context, topology, parallel, mb_shape)
    if batch < pp:
        full_mb_floor = _eval_cached_step_floor(
            eval_context, topology, replace(parallel, pp=1), mb_shape
        )
        return (
            max(
                full_mb_floor.serial_estimate_s,
                float(num_microbatches) * stage_floor.serial_estimate_s,
            ),
            stage_floor.binding,
        )
    return float(pp) * stage_floor.serial_estimate_s, stage_floor.binding


def _prefill_step_floor(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    prefix_hit_rate: float,
) -> StepFloor:
    """Evaluate prompt prefill floor for a given topology, parallel, and hit rate."""
    cache_key = (topology.name, parallel, prefix_hit_rate)
    cached = eval_context.prefill_cache.get(cache_key)
    if cached is not None:
        return cached
    result = step_floor(
        eval_context.geometry,
        eval_context.chip,
        parallel,
        StepShape(
            batch=0,
            rows_per_seq=0,
            context=0,
            state_writes=0,
            prefill_tokens=eval_context.scenario.isl,
            prefix_hit_rate=prefix_hit_rate,
        ),
        FloorOptions(
            dtypes=eval_context.dtypes,
            collective_latency_s=eval_context.collective_latency_s,
            topology=topology,
            activation_model=eval_context.activation_model,
        ),
    )
    eval_context.prefill_cache[cache_key] = result
    return result


def _clamp_resident_batch(
    eval_context: _SolveContext,
    parallel: Parallel,
    b_start: int,
) -> tuple[int, HbmBreakdown] | None:
    """Decrease resident batch until HBM usage fits within hbm_limit_b."""
    resident_batch = b_start
    hbm_bd = hbm_breakdown_per_core_bytes(
        eval_context.geometry,
        eval_context.dtypes,
        parallel,
        SeqKvLayout(resident_batch, eval_context.ctx_full),
        chip=eval_context.chip,
    )
    while resident_batch > 1 and hbm_bd.total > eval_context.hbm_limit_b:
        resident_batch -= 1
        hbm_bd = hbm_breakdown_per_core_bytes(
            eval_context.geometry,
            eval_context.dtypes,
            parallel,
            SeqKvLayout(resident_batch, eval_context.ctx_full),
            chip=eval_context.chip,
        )
    if hbm_bd.total > eval_context.hbm_limit_b:
        return None
    return resident_batch, hbm_bd


def _find_prefill_options(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[_PrefillOption], float]:
    """Enumerate all legal, HBM-feasible prefill configurations, collapsing residual."""
    prefill_options: list[_PrefillOption] = []
    min_req_ttft_ms = math.inf
    for topology in topos:
        if (
            eval_context.prefill_topology is not None
            and topology.name != eval_context.prefill_topology
        ):
            continue
        offload_b_per_core = (
            float(eval_context.scenario.kv_offload_gb)
            * UNIT_GIGA
            / float(max(1, topology.num_cores))
        )
        for base_parallel in _candidate_prefill_parallels(
            eval_context.geometry, topology
        ):
            fitting_opts: list[_PrefillOption] = []
            for parallel in _candidate_residual_parallels(
                eval_context.geometry, base_parallel
            ):
                hbm_bd = hbm_breakdown_per_core_bytes(
                    eval_context.geometry,
                    eval_context.dtypes,
                    parallel,
                    SeqKvLayout(1, eval_context.scenario.isl),
                    chip=eval_context.chip,
                )
                if hbm_bd.total > eval_context.hbm_limit_b:
                    continue
                prefill_floor = _prefill_step_floor(
                    eval_context,
                    topology,
                    parallel,
                    eval_context.scenario.prefix_hit_rate,
                )
                pp_factor = 1.0 + float(parallel.pp - 1) / float(
                    max(1, eval_context.n_chunks)
                )
                ttft_ms = (
                    prefill_floor.serial_estimate_s * pp_factor * UNIT_MILLIS_PER_SECOND
                )
                floor_ttft_ms = prefill_floor.hard_floor_s * UNIT_MILLIS_PER_SECOND
                env_ttft_ms = prefill_floor.envelope_floor_s * UNIT_MILLIS_PER_SECOND
                rate_p = float(max(1, parallel.dp)) / max(
                    _RATE_FLOOR_EPS, prefill_floor.serial_estimate_s
                )
                w_base_p = hbm_footprint_per_core_bytes(
                    eval_context.geometry,
                    eval_context.dtypes,
                    parallel,
                    SeqKvLayout(0, 0),
                    chip=eval_context.chip,
                )
                kv_b_p, fixed_kv_p = kv_bytes_per_core_parts(
                    eval_context.geometry,
                    eval_context.dtypes,
                    parallel,
                    eval_context.ctx_full,
                    physical=False,
                    chip=eval_context.chip,
                )
                fitting_opts.append(
                    _PrefillOption(
                        topology=topology,
                        parallel=parallel,
                        ttft_ms=ttft_ms,
                        floor_ttft_ms=floor_ttft_ms,
                        envelope_ttft_ms=env_ttft_ms,
                        rate_p=rate_p,
                        hbm_bd=hbm_bd,
                        binding=prefill_floor.binding,
                        w_base_b=w_base_p,
                        kv_bytes_per_tok=kv_b_p,
                        fixed_kv_per_seq_b=fixed_kv_p,
                        offload_b_per_core=offload_b_per_core,
                    )
                )
            if not fitting_opts:
                continue
            best_opt = fitting_opts[0]
            for cand_opt in fitting_opts[1:]:
                if cand_opt.ttft_ms < best_opt.ttft_ms - 1e-12:
                    best_opt = cand_opt
            min_mode_floor_ttft_ms = min(opt.floor_ttft_ms for opt in fitting_opts)
            min_mode_env_ttft_ms = min(opt.envelope_ttft_ms for opt in fitting_opts)
            alt_opt = next((opt for opt in fitting_opts if opt is not best_opt), None)
            min_req_ttft_ms = min(min_req_ttft_ms, min_mode_floor_ttft_ms)
            prefill_options.append(
                replace(
                    best_opt,
                    floor_ttft_ms=min_mode_floor_ttft_ms,
                    envelope_ttft_ms=min_mode_env_ttft_ms,
                    alt_sp=alt_opt,
                )
            )
    return prefill_options, min_req_ttft_ms


def _solve_mixed_think_batch(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    concurrency: int,
    prefill_step_floor: StepFloor,
) -> int:
    """Resolve Little's-law active local batch size under non-zero think time."""
    dec_steps_per_req = _decode_steps_per_req(
        eval_context.scenario.osl, eval_context.effective_accept_len
    )
    local_batch = max(1, math.ceil(concurrency / max(1, parallel.dp)))
    think_s = eval_context.scenario.think_time_s
    if think_s is None or think_s <= 0.0:
        return local_batch
    pp_factor = 1.0 + float(parallel.pp - 1) / float(max(1, eval_context.n_chunks))
    for _ in range(2):
        dec_serial_s, _, _ = _eval_pp_decode_cycle(
            eval_context, topology, parallel, local_batch, need_hard_floor=False
        )
        service_time_s = (
            prefill_step_floor.serial_estimate_s * pp_factor
            + dec_steps_per_req * dec_serial_s
        )
        active_concurrency = (
            float(concurrency) * service_time_s / (service_time_s + (think_s or 0.0))
        )
        local_batch = max(1, math.ceil(active_concurrency / max(1, parallel.dp)))
    return local_batch


def _eval_mixed_step_floors(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    resident_batch: int,
    eff_hit_rate: float,
) -> tuple[int, float, str, float, str, float]:
    """Evaluate PP-aware decode and mixed-chunk serial times at effective hit rate."""
    eff_isl_c = max(1, math.ceil((1.0 - eff_hit_rate) * eval_context.scenario.isl))
    cached_prefix_c = max(0, eval_context.scenario.isl - eff_isl_c)
    n_chunks_c = max(
        1, math.ceil(eff_isl_c / max(1, eval_context.scenario.prefill_chunk))
    )
    chunk_tok_c = min(eff_isl_c, eval_context.scenario.prefill_chunk)
    avg_prior = cached_prefix_c + CAUSAL_MIDPOINT_HALF * float(
        max(0, eff_isl_c - chunk_tok_c)
    )
    mix_span = avg_prior + float(chunk_tok_c)
    mix_pre_total = round(mix_span)
    mix_hit_rate = avg_prior / mix_span if mix_span > 0.0 else 0.0
    shared_tok = round(eff_hit_rate * eval_context.scenario.isl)
    decode_serial_s, _, decode_binding = _eval_pp_decode_cycle(
        eval_context,
        topology,
        parallel,
        resident_batch,
        shared_tok,
        need_hard_floor=False,
    )
    if eval_context.spec_k > 0:
        ver_only_serial_s, _ = _eval_pp_step_serial(
            eval_context,
            topology,
            parallel,
            StepShape(
                batch=resident_batch,
                rows_per_seq=eval_context.spec_k + 1,
                context=eval_context.ctx_mid,
                state_writes=1,
                shared_prefix_tokens=shared_tok,
            ),
            resident_batch,
        )
    else:
        ver_only_serial_s = decode_serial_s
    draft_serial_s = max(0.0, decode_serial_s - ver_only_serial_s)
    mix_verify_serial_s, mix_binding = _eval_pp_step_serial(
        eval_context,
        topology,
        parallel,
        StepShape(
            batch=resident_batch,
            rows_per_seq=eval_context.spec_k + 1,
            context=eval_context.ctx_mid,
            state_writes=1,
            prefill_tokens=mix_pre_total,
            prefix_hit_rate=mix_hit_rate,
            shared_prefix_tokens=shared_tok,
        ),
        resident_batch,
    )
    return (
        n_chunks_c,
        decode_serial_s,
        decode_binding,
        mix_verify_serial_s + draft_serial_s,
        mix_binding,
        mix_verify_serial_s,
    )


def _eval_pure_chunk_step_floor(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    hit_rate: float,
) -> StepFloor:
    """Evaluate one pure-prefill tail chunk floor for mixed-mode overflow."""
    eff_isl_c = max(1, math.ceil((1.0 - hit_rate) * eval_context.scenario.isl))
    cached_prefix_c = max(0, eval_context.scenario.isl - eff_isl_c)
    chunk_tok_c = min(eff_isl_c, eval_context.scenario.prefill_chunk)
    avg_prior = cached_prefix_c + CAUSAL_MIDPOINT_HALF * float(
        max(0, eff_isl_c - chunk_tok_c)
    )
    mix_span = avg_prior + float(chunk_tok_c)
    return _eval_cached_step_floor(
        eval_context,
        topology,
        parallel,
        StepShape(
            batch=0,
            rows_per_seq=0,
            context=0,
            state_writes=0,
            prefill_tokens=round(mix_span),
            prefix_hit_rate=avg_prior / mix_span if mix_span > 0.0 else 0.0,
        ),
    )


def _compute_mixed_bounds(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    concurrency: int,
    bounds: _MixedPointBounds,
) -> tuple[float, float, float, StepFloor, float, float, float, float]:
    """Compute mixed-pool capacity ceiling, floor throughput, and latency bounds."""
    replica_batch_cap = max(1, bounds.physical_batch_cap)
    bound_in = _context_to_bound_input(eval_context, topology, parallel)
    capacity_rate_bound = _mixed_capacity_work_bound(bound_in, replica_batch_cap)
    bound_ttft_val = bounds.sf_ttft.hard_floor_s * UNIT_MILLIS_PER_SECOND
    envelope_ttft_val = bounds.sf_ttft.envelope_floor_s * UNIT_MILLIS_PER_SECOND
    think_s = eval_context.scenario.think_time_s
    if think_s is not None:
        dec_toks = float(_decode_tokens(eval_context.scenario.osl))
        dec_b1_s = (
            dec_toks
            * (
                _eval_pp_bound_decode_cycle(bound_in, 1).hard_floor_s
                / eval_context.effective_accept_len
            )
            if dec_toks > 0.0
            else 0.0
        )
        request_rate_limit = min(
            capacity_rate_bound,
            float(concurrency)
            / max(
                _RATE_FLOOR_EPS,
                bound_ttft_val / UNIT_MILLIS_PER_SECOND + dec_b1_s + think_s,
            ),
        )
    else:
        request_rate_limit = capacity_rate_bound
    floor_tput = (
        request_rate_limit
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
    ) / float(topology.num_chips)
    min_decode_seqs, bound_sf_dec = _solve_fixed_point_residency(
        bound_in,
        concurrency,
        replica_batch_cap,
        max(1, parallel.dp),
        capacity_rate_bound,
    )
    bound_tpot_val = (
        bound_sf_dec.hard_floor_s * UNIT_MILLIS_PER_SECOND
    ) / eval_context.effective_accept_len
    envelope_tpot_val = (
        bound_sf_dec.envelope_floor_s * UNIT_MILLIS_PER_SECOND
    ) / eval_context.effective_accept_len
    return (
        capacity_rate_bound,
        floor_tput,
        min_decode_seqs,
        bound_sf_dec,
        bound_tpot_val,
        envelope_tpot_val,
        bound_ttft_val,
        envelope_ttft_val,
    )


def _evaluate_mixed_point(
    eval_context: _SolveContext,
    topology: SliceTopology,
    parallel: Parallel,
    concurrency: int,
    bounds: _MixedPointBounds,
) -> ParetoPoint | None:
    """Evaluate one concurrency target for a single-pool mixed serving configuration."""
    if bounds.b_hbm_max < 1:
        return None
    dec_steps_per_req = _decode_steps_per_req(
        eval_context.scenario.osl, eval_context.effective_accept_len
    )
    offload_b_per_core = (
        float(eval_context.scenario.kv_offload_gb)
        * UNIT_GIGA
        / float(max(1, topology.num_cores))
    )
    local_batch = _solve_mixed_think_batch(
        eval_context, topology, parallel, concurrency, bounds.sf_ttft
    )
    b_start = min(local_batch, bounds.b_hbm_max)
    clamped = _clamp_resident_batch(eval_context, parallel, b_start)
    if clamped is None:
        return None
    resident_batch, hbm_bd = clamped
    working_set_batch = max(1, math.ceil(concurrency / max(1, parallel.dp)))
    cap_tok, eff_hit_rate = kv_capacity_and_hit_rate(
        eval_context.geometry,
        eval_context.dtypes,
        parallel,
        resident_batch,
        KvFitInput(
            working_set_batch=working_set_batch,
            ctx_full=eval_context.ctx_full,
            hbm_limit_b=eval_context.hbm_limit_b,
            w_base_b=bounds.w_base_b,
            b_kv_phys_core=bounds.kv_bytes_per_tok,
            offload_b_per_core=offload_b_per_core,
            prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
            kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
            fixed_kv_per_seq_bytes=bounds.fixed_kv_per_seq_b,
        ),
        chip=eval_context.chip,
    )
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    sf_ttft_c = (
        _prefill_step_floor(eval_context, topology, parallel, eff_hit_rate)
        if eff_hit_rate != eval_context.scenario.prefix_hit_rate
        else bounds.sf_ttft
    )
    (
        n_chunks_c,
        decode_serial_s,
        decode_binding,
        mix_serial_s,
        mix_binding,
        mix_verify_serial_s,
    ) = _eval_mixed_step_floors(
        eval_context, topology, parallel, resident_batch, eff_hit_rate
    )
    pp_factor = 1.0 + float(parallel.pp - 1) / float(max(1, n_chunks_c))
    if dec_steps_per_req <= 0.0:
        sf_pure_chunk = _eval_pure_chunk_step_floor(
            eval_context, topology, parallel, eff_hit_rate
        )
        pure_chunk_serial_s = sf_pure_chunk.serial_estimate_s * pp_factor
        replica_req_time_s = float(n_chunks_c) * pure_chunk_serial_s
        prefill_backlog_s = float(resident_batch * n_chunks_c) * pure_chunk_serial_s
        p_mix = 0.0
        effective_step_s = decode_serial_s
    else:
        dec_slots_per_req = dec_steps_per_req / float(max(1, resident_batch))
        if float(resident_batch * n_chunks_c) <= dec_steps_per_req:
            replica_req_time_s = (
                dec_slots_per_req - float(n_chunks_c)
            ) * decode_serial_s + float(n_chunks_c) * mix_serial_s
            prefill_backlog_s = 0.0
        else:
            sf_pure_chunk = _eval_pure_chunk_step_floor(
                eval_context, topology, parallel, eff_hit_rate
            )
            pure_chunk_serial_s = sf_pure_chunk.serial_estimate_s * pp_factor
            replica_req_time_s = (
                dec_slots_per_req * mix_serial_s
                + (float(n_chunks_c) - dec_slots_per_req) * pure_chunk_serial_s
            )
            prefill_backlog_s = (
                float(resident_batch * n_chunks_c) - dec_steps_per_req
            ) * pure_chunk_serial_s
        p_mix = min(1.0, float(resident_batch * n_chunks_c) / dec_steps_per_req)
        effective_step_s = (1.0 - p_mix) * decode_serial_s + p_mix * mix_serial_s
    total_req_work_s = float(resident_batch) * replica_req_time_s
    res_cap = float(resident_batch * max(1, parallel.dp))
    c_served = min(float(concurrency), res_cap)
    server_cap_rate = res_cap / max(_RATE_FLOOR_EPS, total_req_work_s)
    mix_step_wait_s = mix_verify_serial_s if concurrency > 1 else 0.0
    ttft_in_server_s = max(
        sf_ttft_c.serial_estimate_s * pp_factor + mix_step_wait_s,
        prefill_backlog_s,
    )
    service_time_final_s = ttft_in_server_s + dec_steps_per_req * effective_step_s
    think_s = eval_context.scenario.think_time_s
    if think_s is not None:
        rate_req_s = min(
            server_cap_rate,
            float(concurrency) / max(_RATE_FLOOR_EPS, service_time_final_s + think_s),
        )
    else:
        rate_req_s = c_served / max(_RATE_FLOOR_EPS, total_req_work_s)
    est_tpot_ms = (
        effective_step_s * UNIT_MILLIS_PER_SECOND / eval_context.effective_accept_len
    )
    c_demand = min(float(concurrency), float(local_batch * max(1, parallel.dp)))
    queue_wait_s = (
        max(0.0, (c_demand / max(_RATE_FLOOR_EPS, res_cap)) - 1.0)
        * service_time_final_s
    )
    est_ttft_ms = (ttft_in_server_s + queue_wait_s) * UNIT_MILLIS_PER_SECOND
    interactivity = UNIT_MILLIS_PER_SECOND / max(1e-6, est_tpot_ms)
    out_tput = rate_req_s * float(eval_context.scenario.osl) / float(topology.num_chips)
    tot_tput = (
        rate_req_s
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
        / float(topology.num_chips)
    )
    (
        capacity_rate_bound,
        floor_tput,
        min_decode_seqs,
        bound_sf_dec,
        bound_tpot_val,
        envelope_tpot_val,
        bound_ttft_val,
        envelope_ttft_val,
    ) = _compute_mixed_bounds(eval_context, topology, parallel, concurrency, bounds)
    use_env = eval_context.activation_model.prunes_on_envelope
    verdict = _resolve_slo_verdict(
        eval_context.scenario,
        est_ttft_ms,
        est_tpot_ms,
        envelope_ttft_val if use_env else bound_ttft_val,
        envelope_tpot_val if use_env else bound_tpot_val,
    )
    verdict = _apply_capacity_ceiling_verdict(
        eval_context.scenario,
        verdict,
        concurrency=concurrency,
        pool_capacity_req_s=capacity_rate_bound,
    )
    pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="mixed",
            instances=1,
            topology=topology,
            parallel=parallel,
            hbm_bd=hbm_bd,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            resident_batch_cap=resident_batch,
        )
    )
    binding = (
        f"dec:{decode_binding},mix:{mix_binding}" if p_mix > 0.0 else decode_binding
    )
    return ParetoPoint(
        mode="mixed",
        concurrency=concurrency,
        total_chips=topology.num_chips,
        prefill_pool=None,
        decode_pool=pool,
        est_ttft_ms=est_ttft_ms,
        est_tpot_ms=est_tpot_ms,
        est_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=p_mix,
        binding=binding,
        slo_met=verdict == SLO_VERDICT_ESTIMATE_MEETS,
        bound_tpot_ms=bound_tpot_val,
        bound_ttft_ms=bound_ttft_val,
        slo_verdict=verdict,
        decode_floor=_floor_times(bound_sf_dec),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=eff_hit_rate,
        min_decode_seqs=min_decode_seqs,
        floor_throughput_per_chip=floor_tput,
        envelope_tpot_ms=envelope_tpot_val,
        envelope_ttft_ms=envelope_ttft_val,
        envelope_decode_seqs=min_decode_seqs,
    )


def _decode_capacity_profile(
    eval_context: _SolveContext,
    parallel: Parallel,
) -> tuple[float, int, float, float, float]:
    """Compute base weights, HBM batch limit, and logical KV byte components."""
    w_base_b = hbm_footprint_per_core_bytes(
        eval_context.geometry,
        eval_context.dtypes,
        parallel,
        SeqKvLayout(0, 0),
        chip=eval_context.chip,
    )
    hbm_c1 = hbm_footprint_per_core_bytes(
        eval_context.geometry,
        eval_context.dtypes,
        parallel,
        SeqKvLayout(1, eval_context.ctx_full),
        chip=eval_context.chip,
    )
    per_seq_b = hbm_c1 - w_base_b
    b_hbm_max = (
        int((eval_context.hbm_limit_b - w_base_b) // per_seq_b)
        if (hbm_c1 <= eval_context.hbm_limit_b and per_seq_b > 0.0)
        else (eval_context.c_max if hbm_c1 <= eval_context.hbm_limit_b else 0)
    )
    kv_bytes_core, fixed_kv_per_seq_b = kv_bytes_per_core_parts(
        eval_context.geometry,
        eval_context.dtypes,
        parallel,
        eval_context.ctx_full,
        physical=False,
        chip=eval_context.chip,
    )
    return w_base_b, b_hbm_max, kv_bytes_core, fixed_kv_per_seq_b, hbm_c1


def _evaluate_mixed_candidates(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[ParetoPoint], float, float]:
    """Evaluate all single-pool mixed candidate parallelisms, collapsing residual."""
    candidates: list[ParetoPoint] = []
    min_c1_tpot_ms = math.inf
    min_req_ttft_ms = math.inf
    for topology in topos:
        if (
            eval_context.decode_topology is not None
            and topology.name != eval_context.decode_topology
        ):
            continue
        for base_parallel in _candidate_decode_parallels(
            eval_context.geometry, topology
        ):
            if eval_context.mode in ("auto", "mixed"):
                eval_context.enumerated += len(eval_context.scenario.concurrencies)
            fitting_modes: list[tuple[Parallel, _MixedPointBounds]] = []
            for parallel in _candidate_residual_parallels(
                eval_context.geometry, base_parallel
            ):
                (
                    w_base_b,
                    b_hbm_max,
                    kv_bytes_core,
                    fixed_kv_per_seq_b,
                    hbm_c1,
                ) = _decode_capacity_profile(eval_context, parallel)
                if hbm_c1 <= eval_context.hbm_limit_b:
                    _, c1_hard_s, _ = _eval_pp_decode_cycle(
                        eval_context, topology, parallel, 1, need_hard_floor=True
                    )
                    min_c1_tpot_ms = min(
                        min_c1_tpot_ms,
                        c1_hard_s
                        * UNIT_MILLIS_PER_SECOND
                        / eval_context.effective_accept_len,
                    )
                if eval_context.mode not in ("auto", "mixed") or b_hbm_max < 1:
                    continue
                prefill_step_floor = _prefill_step_floor(
                    eval_context,
                    topology,
                    parallel,
                    eval_context.scenario.prefix_hit_rate,
                )
                min_req_ttft_ms = min(
                    min_req_ttft_ms,
                    prefill_step_floor.hard_floor_s * UNIT_MILLIS_PER_SECOND,
                )
                physical_batch_cap = max(
                    b_hbm_max, _steady_state_batch_cap_phys(eval_context, parallel)
                )
                fitting_modes.append(
                    (
                        parallel,
                        _MixedPointBounds(
                            w_base_b=w_base_b,
                            kv_bytes_per_tok=kv_bytes_core,
                            b_hbm_max=b_hbm_max,
                            physical_batch_cap=physical_batch_cap,
                            sf_ttft=prefill_step_floor,
                            fixed_kv_per_seq_b=fixed_kv_per_seq_b,
                        ),
                    )
                )
            if not fitting_modes:
                continue
            use_env = eval_context.activation_model.prunes_on_envelope
            for concurrency in eval_context.scenario.concurrencies:
                mode_points: list[ParetoPoint] = []
                for parallel, bounds in fitting_modes:
                    point = _evaluate_mixed_point(
                        eval_context, topology, parallel, concurrency, bounds
                    )
                    if point is not None:
                        mode_points.append(point)
                if not mode_points:
                    continue
                best_point = mode_points[0]
                for point in mode_points[1:]:
                    if _is_strictly_better_point(
                        point, best_point, eval_context.sort_by
                    ):
                        best_point = point
                merged_point = best_point
                for other_point in mode_points:
                    if other_point is not best_point:
                        merged_point = _merge_collapsed_point_bounds(
                            merged_point,
                            other_point,
                            eval_context.scenario,
                            use_envelope=use_env,
                        )
                eval_context.hbm_fit += 1
                candidates.append(merged_point)
    return candidates, min_c1_tpot_ms, min_req_ttft_ms


def _build_prefill_pool_records(
    eval_context: _SolveContext,
    prefill_options: list[_PrefillOption],
) -> list[PoolFloorRecord]:
    """Materialize per-concurrency PoolFloorRecord entries for prefill pools."""
    if eval_context.mode not in ("auto", "disagg"):
        return []
    tot_tokens = float(eval_context.scenario.isl + eval_context.scenario.osl)
    records: list[PoolFloorRecord] = []
    for opt in prefill_options:
        slo_ok = (
            eval_context.scenario.ttft_slo_ms is None
            or opt.ttft_ms <= eval_context.scenario.ttft_slo_ms
        )
        pool = _build_pool_mesh_plan(
            _PoolPlanInput(
                role="prefill",
                instances=1,
                topology=opt.topology,
                parallel=opt.parallel,
                hbm_bd=opt.hbm_bd,
                hbm_limit_gb=eval_context.hbm_limit_gb,
                resident_batch_cap=None,
            )
        )
        tok_tput = opt.rate_p * tot_tokens / float(opt.topology.num_chips)
        records.extend(
            PoolFloorRecord(
                role="prefill",
                concurrency=concurrency,
                pool=pool,
                ttft_ms=opt.ttft_ms,
                ttft_floor_ms=opt.floor_ttft_ms,
                tpot_ms=None,
                tpot_floor_ms=None,
                interactivity_tok_s=None,
                token_throughput_per_chip=tok_tput,
                request_rate_per_instance=opt.rate_p,
                binding=opt.binding,
                slo_met=slo_ok,
            )
            for concurrency in eval_context.scenario.concurrencies
        )
    return records


def _eval_single_decode_pool_record(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    concurrency: int,
) -> PoolFloorRecord | None:
    """Evaluate one standalone decode pool state (single residual mode) at c."""
    dec_steps = _decode_steps_per_req(
        eval_context.scenario.osl, eval_context.effective_accept_len
    )
    if dec_steps <= 0.0:
        return None
    local_batch = max(1, math.ceil(concurrency / max(1, dec.parallel.dp)))
    b_start = min(local_batch, dec.b_hbm_max)
    clamped = _clamp_resident_batch(eval_context, dec.parallel, b_start)
    if clamped is None:
        return None
    resident_batch, hbm_bd = clamped
    working_set_batch = max(1, math.ceil(concurrency / max(1, dec.parallel.dp)))
    cap_tok, eff_hit_rate = kv_capacity_and_hit_rate(
        eval_context.geometry,
        eval_context.dtypes,
        dec.parallel,
        resident_batch,
        KvFitInput(
            working_set_batch=working_set_batch,
            ctx_full=eval_context.ctx_full,
            hbm_limit_b=eval_context.hbm_limit_b,
            w_base_b=dec.w_base_b,
            b_kv_phys_core=dec.kv_bytes_per_tok,
            offload_b_per_core=dec.offload_b_per_core,
            prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
            kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
            fixed_kv_per_seq_bytes=dec.fixed_kv_per_seq_b,
        ),
        chip=eval_context.chip,
    )
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    shared_tok = round(eff_hit_rate * eval_context.scenario.isl)
    decode_serial_s, decode_hard_s, decode_binding = _eval_pp_decode_cycle(
        eval_context,
        dec.topology,
        dec.parallel,
        resident_batch,
        shared_tok,
        need_hard_floor=True,
    )
    tpot_ms = (
        decode_serial_s * UNIT_MILLIS_PER_SECOND / eval_context.effective_accept_len
    )
    tpot_floor_ms = (
        decode_hard_s * UNIT_MILLIS_PER_SECOND / eval_context.effective_accept_len
    )
    interactivity = UNIT_MILLIS_PER_SECOND / max(1e-6, tpot_ms)
    res_cap = float(resident_batch * max(1, dec.parallel.dp))
    c_served = min(float(concurrency), res_cap)
    rate_d = c_served / (dec_steps * max(1e-9, decode_serial_s))
    tot_tokens = float(eval_context.scenario.isl + eval_context.scenario.osl)
    tok_tput = rate_d * tot_tokens / float(dec.topology.num_chips)
    slo_ok = (
        eval_context.scenario.tpot_slo_ms is None
        or tpot_ms <= eval_context.scenario.tpot_slo_ms
    )
    pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="decode",
            instances=1,
            topology=dec.topology,
            parallel=dec.parallel,
            hbm_bd=hbm_bd,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            resident_batch_cap=resident_batch,
        )
    )
    return PoolFloorRecord(
        role="decode",
        concurrency=concurrency,
        pool=pool,
        ttft_ms=None,
        ttft_floor_ms=None,
        tpot_ms=tpot_ms,
        tpot_floor_ms=tpot_floor_ms,
        interactivity_tok_s=interactivity,
        token_throughput_per_chip=tok_tput,
        request_rate_per_instance=rate_d,
        binding=decode_binding,
        slo_met=slo_ok,
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=eff_hit_rate,
    )


def _eval_decode_pool_record(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    concurrency: int,
) -> PoolFloorRecord | None:
    """Evaluate standalone decode pool at c, collapsing legal residual modes."""
    rep_record = _eval_single_decode_pool_record(eval_context, dec, concurrency)
    if dec.alt_sp is None:
        return rep_record
    sp_record = _eval_single_decode_pool_record(eval_context, dec.alt_sp, concurrency)
    if rep_record is None:
        return sp_record
    if sp_record is None:
        return rep_record
    min_tpot_floor = min(
        val
        for val in (rep_record.tpot_floor_ms, sp_record.tpot_floor_ms)
        if val is not None
    )
    winner = (
        sp_record
        if _is_strictly_better_decode_record(
            sp_record, rep_record, eval_context.sort_by
        )
        else rep_record
    )
    return replace(winner, tpot_floor_ms=min_tpot_floor)


def _build_decode_pool_states(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> list[_DisaggDecodeState]:
    """Collect all HBM-feasible decode pool states, bundling legal SP alternative."""
    states: list[_DisaggDecodeState] = []
    for topo_d in topos:
        if (
            eval_context.decode_topology is not None
            and topo_d.name != eval_context.decode_topology
        ):
            continue
        offload_b = (
            float(eval_context.scenario.kv_offload_gb)
            * UNIT_GIGA
            / float(max(1, topo_d.num_cores))
        )
        for base_par_d in _candidate_decode_parallels(eval_context.geometry, topo_d):
            fitting: list[_DisaggDecodeState] = []
            for par_d in _candidate_residual_parallels(
                eval_context.geometry, base_par_d
            ):
                w_base_d, b_hbm_max_d, kv_b_d, fixed_kv_d, _ = _decode_capacity_profile(
                    eval_context, par_d
                )
                if b_hbm_max_d < 1:
                    continue
                fitting.append(
                    _DisaggDecodeState(
                        topology=topo_d,
                        parallel=par_d,
                        w_base_b=w_base_d,
                        b_hbm_max=b_hbm_max_d,
                        kv_bytes_per_tok=kv_b_d,
                        physical_batch_cap=max(
                            b_hbm_max_d,
                            _steady_state_batch_cap_phys(eval_context, par_d),
                        ),
                        offload_b_per_core=offload_b,
                        fixed_kv_per_seq_b=fixed_kv_d,
                    )
                )
            if not fitting:
                continue
            primary = fitting[0]
            if len(fitting) > 1:
                primary = replace(primary, alt_sp=fitting[1])
            states.append(primary)
    return states


def _build_decode_pool_records(
    eval_context: _SolveContext,
    decode_states: list[_DisaggDecodeState],
) -> list[PoolFloorRecord]:
    """Materialize per-concurrency PoolFloorRecord entries for decode pools."""
    if eval_context.mode not in ("auto", "disagg"):
        return []
    records: list[PoolFloorRecord] = []
    for dec in decode_states:
        for concurrency in eval_context.scenario.concurrencies:
            record = _eval_decode_pool_record(eval_context, dec, concurrency)
            if record is not None:
                records.append(record)
    return records


def _compute_disagg_metrics(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    target: _DisaggRatioTarget,
    batch_stats: _DisaggBatchStats,
    decode_serial_s: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Compute latencies, throughputs, coupled rate, and decode cap for disagg."""
    dec_steps_per_req = _decode_steps_per_req(
        eval_context.scenario.osl, eval_context.effective_accept_len
    )
    est_tpot_ms = (
        decode_serial_s * UNIT_MILLIS_PER_SECOND / eval_context.effective_accept_len
    )
    interactivity = UNIT_MILLIS_PER_SECOND / max(1e-6, est_tpot_ms)
    res_cap_d = float(
        batch_stats.resident_batch * max(1, target.decode_instances * dec.parallel.dp)
    )
    c_served_d = min(float(target.concurrency), res_cap_d)
    if dec_steps_per_req <= 0.0:
        rate_d_cap = math.inf
        queue_wait_d_ms = 0.0
    else:
        rate_d_cap = c_served_d / (
            dec_steps_per_req * max(_RATE_FLOOR_EPS, decode_serial_s)
        )
        c_demand_d = min(
            float(target.concurrency),
            float(
                batch_stats.local_batch
                * max(1, target.decode_instances * dec.parallel.dp)
            ),
        )
        queue_wait_d_ms = (
            max(0.0, (c_demand_d / max(_RATE_FLOOR_EPS, res_cap_d)) - 1.0)
            * (dec_steps_per_req * decode_serial_s)
            * UNIT_MILLIS_PER_SECOND
        )
    prefill_rate = float(target.prefill_instances) * batch_stats.rate_p_eff
    think_s = eval_context.scenario.think_time_s
    if think_s is not None:
        client_rate = float(target.concurrency) / max(
            _RATE_FLOOR_EPS,
            batch_stats.ttft_eff_ms / UNIT_MILLIS_PER_SECOND
            + dec_steps_per_req * decode_serial_s
            + think_s,
        )
        coupled_rate = min(prefill_rate, rate_d_cap, client_rate)
    else:
        coupled_rate = min(prefill_rate, rate_d_cap)
    ll_ttft_s = max(
        0.0,
        float(target.concurrency) / max(_RATE_FLOOR_EPS, coupled_rate)
        - dec_steps_per_req * decode_serial_s
        - (think_s or 0.0),
    )
    est_ttft_d_ms = max(
        batch_stats.ttft_eff_ms + queue_wait_d_ms,
        ll_ttft_s * UNIT_MILLIS_PER_SECOND,
    )
    out_tput = (
        coupled_rate * float(eval_context.scenario.osl) / float(target.total_chips)
    )
    tot_tput = (
        coupled_rate
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
        / float(target.total_chips)
    )
    return (
        est_ttft_d_ms,
        est_tpot_ms,
        interactivity,
        out_tput,
        tot_tput,
        coupled_rate,
        rate_d_cap,
    )


def _solve_disagg_think_batch(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
) -> int:
    """Resolve Little's-law active local batch for disaggregated decode pool."""
    dec_steps_per_req = _decode_steps_per_req(
        eval_context.scenario.osl, eval_context.effective_accept_len
    )
    local_batch = max(
        1,
        math.ceil(
            target.concurrency / max(1, target.decode_instances * dec.parallel.dp)
        ),
    )
    think_s = eval_context.scenario.think_time_s
    if think_s is None or think_s <= 0.0:
        return local_batch
    for _ in range(2):
        dec_serial_s, _, _ = _eval_pp_decode_cycle(
            eval_context, dec.topology, dec.parallel, local_batch, need_hard_floor=False
        )
        service_time_s = (
            pref.ttft_ms / UNIT_MILLIS_PER_SECOND + dec_steps_per_req * dec_serial_s
        )
        active_concurrency = (
            float(target.concurrency)
            * service_time_s
            / (service_time_s + (think_s or 0.0))
        )
        local_batch = max(
            1,
            math.ceil(
                active_concurrency / max(1, target.decode_instances * dec.parallel.dp)
            ),
        )
    return local_batch


def _resolve_effective_prefill(
    eval_context: _SolveContext,
    pref: _PrefillOption,
    eff_hit_rate: float,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> tuple[float, float]:
    """Recompute cached prefill (ttft_ms, rate_p) when prefix hit rate is clamped."""
    if eff_hit_rate >= eval_context.scenario.prefix_hit_rate:
        return pref.ttft_ms, pref.rate_p
    p_key = (pref.topology.name, pref.parallel, round(eff_hit_rate, 6))
    cached_p = cache.get(p_key)
    if cached_p is not None:
        return cached_p
    eff_isl_p = max(1, math.ceil((1.0 - eff_hit_rate) * eval_context.scenario.isl))
    n_chunks_p = max(
        1, math.ceil(eff_isl_p / max(1, eval_context.scenario.prefill_chunk))
    )
    sf_p_eff = _prefill_step_floor(
        eval_context, pref.topology, pref.parallel, eff_hit_rate
    )
    ttft_p_eff = (
        sf_p_eff.serial_estimate_s
        * (1.0 + float(pref.parallel.pp - 1) / float(max(1, n_chunks_p)))
        * UNIT_MILLIS_PER_SECOND
    )
    rate_p_eff = float(max(1, pref.parallel.dp)) / max(1e-9, sf_p_eff.serial_estimate_s)
    cached_p = (ttft_p_eff, rate_p_eff)
    cache[p_key] = cached_p
    return cached_p


def _prefill_unit_capacity_rate(
    eval_context: _SolveContext,
    pref: _PrefillOption,
) -> float:
    """Return cached per-instance prefill capacity ceiling (req/s) across residual."""
    cache_key = (pref.topology.name, pref.parallel)
    cached = eval_context.prefill_unit_cap_cache.get(cache_key)
    if cached is not None:
        return cached
    bound_in = _context_to_bound_input(eval_context, pref.topology, pref.parallel)

    def _one_rate(opt: _PrefillOption) -> float:
        pre_hbm, pre_mm, pre_vec, pre_load = _eval_unpadded_prefill_buckets(
            bound_in, opt.parallel, opt.topology, zero_weights=True
        )
        pre_coll = eval_link_subset_bound_s(pre_load, eval_context.chip, opt.topology)
        pre_time = max(pre_hbm, pre_mm, pre_vec, pre_coll)
        return (
            float(max(1, opt.parallel.dp)) / max(_CAP_DENOM_EPS, pre_time)
            if pre_time > 0.0
            else math.inf
        )

    rate = _one_rate(pref)
    if pref.alt_sp is not None:
        rate = max(rate, _one_rate(pref.alt_sp))
    eval_context.prefill_unit_cap_cache[cache_key] = rate
    return rate


def _decode_unit_capacity_rate(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    replica_batch_cap: int,
) -> float:
    """Return cached per-instance decode capacity ceiling (req/s)."""
    cache_key = (dec.topology.name, dec.parallel, replica_batch_cap)
    cached = eval_context.decode_unit_cap_cache.get(cache_key)
    if cached is not None:
        return cached
    dec_steps = _decode_steps_per_req(
        eval_context.scenario.osl, eval_context.effective_accept_len
    )
    if dec_steps <= 0.0:
        eval_context.decode_unit_cap_cache[cache_key] = math.inf
        return math.inf
    bound_in = _context_to_bound_input(eval_context, dec.topology, dec.parallel)
    dec_hbm, dec_mm, dec_vec, dec_load = _eval_unpadded_decode_buckets(
        bound_in, replica_batch_cap
    )
    dec_coll = eval_link_subset_bound_s(dec_load, eval_context.chip, dec.topology)
    dec_scale = dec_steps / float(max(1, replica_batch_cap))
    dec_time = dec_scale * max(dec_hbm, dec_mm, dec_vec, dec_coll)
    rate = float(max(1, dec.parallel.dp)) / max(_CAP_DENOM_EPS, dec_time)
    eval_context.decode_unit_cap_cache[cache_key] = rate
    return rate


def _evaluate_single_disagg_pair(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> ParetoPoint | None:
    """Evaluate one instance ratio pair (x_p, y_d) for a single decode residual mode."""
    if (
        eval_context.max_chips is not None
        and target.total_chips > eval_context.max_chips
    ):
        return None
    if dec.b_hbm_max < 1:
        return None
    local_batch = _solve_disagg_think_batch(eval_context, dec, pref, target)
    b_start = min(local_batch, dec.b_hbm_max)
    clamp_key = (dec.parallel, b_start)
    if clamp_key in eval_context.clamp_cache:
        clamped = eval_context.clamp_cache[clamp_key]
    else:
        clamped = _clamp_resident_batch(eval_context, dec.parallel, b_start)
        eval_context.clamp_cache[clamp_key] = clamped
    if clamped is None:
        return None
    resident_batch, hbm_bd = clamped
    working_set_decode = max(
        1,
        math.ceil(
            target.concurrency / max(1, target.decode_instances * dec.parallel.dp)
        ),
    )
    dec_kv_key = (
        dec.parallel,
        resident_batch,
        working_set_decode,
        dec.offload_b_per_core,
    )
    cached_dec_kv = eval_context.kv_fit_cache.get(dec_kv_key)
    if cached_dec_kv is not None:
        cap_tok, eff_hit_decode = cached_dec_kv
    else:
        cap_tok, eff_hit_decode = kv_capacity_and_hit_rate(
            eval_context.geometry,
            eval_context.dtypes,
            dec.parallel,
            resident_batch,
            KvFitInput(
                working_set_batch=working_set_decode,
                ctx_full=eval_context.ctx_full,
                hbm_limit_b=eval_context.hbm_limit_b,
                w_base_b=dec.w_base_b,
                b_kv_phys_core=dec.kv_bytes_per_tok,
                offload_b_per_core=dec.offload_b_per_core,
                prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
                kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
                fixed_kv_per_seq_bytes=dec.fixed_kv_per_seq_b,
            ),
            chip=eval_context.chip,
        )
        eval_context.kv_fit_cache[dec_kv_key] = (cap_tok, eff_hit_decode)
    working_set_prefill = max(
        1,
        math.ceil(
            target.concurrency / max(1, target.prefill_instances * pref.parallel.dp)
        ),
    )
    pref_kv_key = (
        pref.parallel,
        1,
        working_set_prefill,
        pref.offload_b_per_core,
    )
    cached_pref_kv = eval_context.kv_fit_cache.get(pref_kv_key)
    if cached_pref_kv is not None:
        _, eff_hit_prefill = cached_pref_kv
    else:
        pref_cap_tok, eff_hit_prefill = kv_capacity_and_hit_rate(
            eval_context.geometry,
            eval_context.dtypes,
            pref.parallel,
            1,
            KvFitInput(
                working_set_batch=working_set_prefill,
                ctx_full=eval_context.ctx_full,
                hbm_limit_b=eval_context.hbm_limit_b,
                w_base_b=pref.w_base_b,
                b_kv_phys_core=pref.kv_bytes_per_tok,
                offload_b_per_core=pref.offload_b_per_core,
                prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
                kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
                fixed_kv_per_seq_bytes=pref.fixed_kv_per_seq_b,
            ),
            chip=eval_context.chip,
        )
        eval_context.kv_fit_cache[pref_kv_key] = (pref_cap_tok, eff_hit_prefill)
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    ttft_eff_ms, rate_p_eff = _resolve_effective_prefill(
        eval_context, pref, eff_hit_prefill, cache
    )
    shared_tok = round(eff_hit_decode * eval_context.scenario.isl)
    decode_serial_s, _, decode_binding = _eval_pp_decode_cycle(
        eval_context,
        dec.topology,
        dec.parallel,
        resident_batch,
        shared_tok,
        need_hard_floor=False,
    )
    (
        est_ttft_d_ms,
        est_tpot_ms,
        interactivity,
        out_tput,
        tot_tput,
        coupled_rate,
        rate_d_cap,
    ) = _compute_disagg_metrics(
        eval_context,
        dec,
        target,
        _DisaggBatchStats(resident_batch, local_batch, ttft_eff_ms, rate_p_eff),
        decode_serial_s,
    )
    # One fixed-point decode occupancy pass when prefill starves decode.
    dec_steps_per_req = _decode_steps_per_req(
        eval_context.scenario.osl, eval_context.effective_accept_len
    )
    if dec_steps_per_req > 0.0 and coupled_rate < rate_d_cap:
        rb_eff = min(
            resident_batch,
            max(
                1,
                math.ceil(
                    coupled_rate
                    * dec_steps_per_req
                    * decode_serial_s
                    / float(max(1, target.decode_instances * dec.parallel.dp))
                ),
            ),
        )
        if rb_eff < resident_batch:
            decode_serial_s, _, decode_binding = _eval_pp_decode_cycle(
                eval_context,
                dec.topology,
                dec.parallel,
                rb_eff,
                shared_tok,
                need_hard_floor=False,
            )
            (
                est_ttft_d_ms,
                est_tpot_ms,
                interactivity,
                out_tput,
                tot_tput,
                _,
                _,
            ) = _compute_disagg_metrics(
                eval_context,
                dec,
                target,
                _DisaggBatchStats(resident_batch, local_batch, ttft_eff_ms, rate_p_eff),
                decode_serial_s,
            )
    replica_batch_cap = max(1, dec.physical_batch_cap)
    pref_cap_rate = float(target.prefill_instances) * _prefill_unit_capacity_rate(
        eval_context, pref
    )
    dec_cap_rate = float(target.decode_instances) * _decode_unit_capacity_rate(
        eval_context, dec, replica_batch_cap
    )
    capacity_rate_bound = min(pref_cap_rate, dec_cap_rate)
    bound_ttft_val = pref.floor_ttft_ms
    envelope_ttft_val = pref.envelope_ttft_ms
    b1_key = (dec.topology.name, dec.parallel, 1)
    bound_sf_b1 = eval_context.bound_decode_cache.get(b1_key)
    if bound_sf_b1 is None:
        bound_in_init = _context_to_bound_input(
            eval_context, dec.topology, dec.parallel
        )
        bound_sf_b1 = _eval_pp_bound_decode_cycle(bound_in_init, 1)
        eval_context.bound_decode_cache[b1_key] = bound_sf_b1
    think_s = eval_context.scenario.think_time_s
    dec_toks = float(_decode_tokens(eval_context.scenario.osl))
    if think_s is not None:
        dec_b1_s = (
            dec_toks * (bound_sf_b1.hard_floor_s / eval_context.effective_accept_len)
            if dec_toks > 0.0
            else 0.0
        )
        request_rate_limit = min(
            capacity_rate_bound,
            float(target.concurrency)
            / max(
                _RATE_FLOOR_EPS,
                bound_ttft_val / UNIT_MILLIS_PER_SECOND + dec_b1_s + think_s,
            ),
        )
    else:
        request_rate_limit = capacity_rate_bound
    floor_tput = (
        request_rate_limit
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
    ) / float(target.total_chips)
    if eval_context.scenario.ttft_slo_ms is None or think_s is None or dec_toks <= 0.0:
        min_decode_seqs = 1.0
        bound_sf_d = bound_sf_b1
    else:
        bound_in = _context_to_bound_input(eval_context, dec.topology, dec.parallel)
        rate_p_floor = float(max(1, target.prefill_instances * pref.parallel.dp)) / max(
            _RATE_FLOOR_EPS, bound_ttft_val / UNIT_MILLIS_PER_SECOND
        )
        min_decode_seqs, bound_sf_d = _solve_fixed_point_residency(
            bound_in,
            target.concurrency,
            replica_batch_cap,
            max(1, target.decode_instances * dec.parallel.dp),
            min(capacity_rate_bound, rate_p_floor),
        )
    bound_tpot_val = (
        bound_sf_d.hard_floor_s * UNIT_MILLIS_PER_SECOND
    ) / eval_context.effective_accept_len
    envelope_tpot_val = (
        bound_sf_d.envelope_floor_s * UNIT_MILLIS_PER_SECOND
    ) / eval_context.effective_accept_len
    use_env = eval_context.activation_model.prunes_on_envelope
    verdict = _resolve_slo_verdict(
        eval_context.scenario,
        est_ttft_d_ms,
        est_tpot_ms,
        envelope_ttft_val if use_env else bound_ttft_val,
        envelope_tpot_val if use_env else bound_tpot_val,
    )
    verdict = _apply_capacity_ceiling_verdict(
        eval_context.scenario,
        verdict,
        concurrency=target.concurrency,
        pool_capacity_req_s=capacity_rate_bound,
    )
    p_pool_key = (
        "prefill",
        target.prefill_instances,
        pref.topology.name,
        pref.parallel,
        None,
    )
    p_pool = eval_context.pool_plan_cache.get(p_pool_key)
    if p_pool is None:
        p_pool = _build_pool_mesh_plan(
            _PoolPlanInput(
                role="prefill",
                instances=target.prefill_instances,
                topology=pref.topology,
                parallel=pref.parallel,
                hbm_bd=pref.hbm_bd,
                hbm_limit_gb=eval_context.hbm_limit_gb,
                resident_batch_cap=None,
            )
        )
        eval_context.pool_plan_cache[p_pool_key] = p_pool
    d_pool_key = (
        "decode",
        target.decode_instances,
        dec.topology.name,
        dec.parallel,
        resident_batch,
    )
    d_pool = eval_context.pool_plan_cache.get(d_pool_key)
    if d_pool is None:
        d_pool = _build_pool_mesh_plan(
            _PoolPlanInput(
                role="decode",
                instances=target.decode_instances,
                topology=dec.topology,
                parallel=dec.parallel,
                hbm_bd=hbm_bd,
                hbm_limit_gb=eval_context.hbm_limit_gb,
                resident_batch_cap=resident_batch,
            )
        )
        eval_context.pool_plan_cache[d_pool_key] = d_pool
    return ParetoPoint(
        mode="disagg",
        concurrency=target.concurrency,
        total_chips=target.total_chips,
        prefill_pool=p_pool,
        decode_pool=d_pool,
        est_ttft_ms=est_ttft_d_ms,
        est_tpot_ms=est_tpot_ms,
        est_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=0.0,
        binding=decode_binding,
        slo_met=verdict == SLO_VERDICT_ESTIMATE_MEETS,
        bound_tpot_ms=bound_tpot_val,
        bound_ttft_ms=bound_ttft_val,
        slo_verdict=verdict,
        decode_floor=_floor_times(bound_sf_d),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=eff_hit_prefill,
        min_decode_seqs=min_decode_seqs,
        floor_throughput_per_chip=floor_tput,
        envelope_tpot_ms=envelope_tpot_val,
        envelope_ttft_ms=envelope_ttft_val,
        envelope_decode_seqs=min_decode_seqs,
    )


def _evaluate_disagg_instance_pair(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> ParetoPoint | None:
    """Evaluate one instance ratio pair (x_p, y_d), collapsing decode residual mode."""
    rep_point = _evaluate_single_disagg_pair(eval_context, dec, pref, target, cache)
    if dec.alt_sp is None:
        if rep_point is not None:
            eval_context.hbm_fit += 1
        return rep_point
    sp_point = _evaluate_single_disagg_pair(
        eval_context, dec.alt_sp, pref, target, cache
    )
    if rep_point is None:
        if sp_point is not None:
            eval_context.hbm_fit += 1
        return sp_point
    if sp_point is None:
        eval_context.hbm_fit += 1
        return rep_point
    eval_context.hbm_fit += 1
    chosen = (
        sp_point
        if _is_strictly_better_point(sp_point, rep_point, eval_context.sort_by)
        else rep_point
    )
    other = rep_point if chosen is sp_point else sp_point
    return _merge_collapsed_point_bounds(
        chosen,
        other,
        eval_context.scenario,
        use_envelope=eval_context.activation_model.prunes_on_envelope,
    )


def _search_disagg_instance_ratios(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    concurrency: int,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> list[ParetoPoint]:
    """Search instance multiplier ratios (x_p, y_d) for one prefill-decode pair."""
    if (
        eval_context.max_chips is not None
        and pref.topology.num_chips + dec.topology.num_chips > eval_context.max_chips
    ):
        return []
    dec_steps_per_req = _decode_steps_per_req(
        eval_context.scenario.osl, eval_context.effective_accept_len
    )
    b_one = math.ceil(concurrency / max(1, dec.parallel.dp))
    dec_one_serial_s, _, _ = _eval_pp_decode_cycle(
        eval_context, dec.topology, dec.parallel, b_one, need_hard_floor=False
    )
    rate_d_one = (
        float(concurrency)
        / (dec_steps_per_req * max(_RATE_FLOOR_EPS, dec_one_serial_s))
        if dec_steps_per_req > 0.0
        else math.inf
    )
    max_xp_override: int | None = None
    max_yd_override: int | None = None
    if eval_context.max_chips is None:
        eff_max_inst = (
            eval_context.max_instances
            if eval_context.max_instances is not None
            else DEFAULT_MAX_INSTANCES
        )
        if math.isfinite(rate_d_one):
            ratio_xp = math.ceil(rate_d_one / max(1e-9, pref.rate_p)) + 1
            ratio_yd = math.ceil(pref.rate_p / max(1e-9, rate_d_one)) + 1
        else:
            ratio_xp = eff_max_inst
            ratio_yd = 1
        max_xp_override = max(1, ratio_xp)
        max_yd_override = max(1, ratio_yd)
    ratio_pairs = _iter_disagg_ratio_pairs(
        eval_context,
        pref.topology,
        dec.topology,
        max_xp_override=max_xp_override,
        max_yd_override=max_yd_override,
    )
    points: list[ParetoPoint] = []
    for prefill_instances, decode_instances, total_chips in ratio_pairs:
        point = _evaluate_disagg_instance_pair(
            eval_context,
            dec,
            pref,
            _DisaggRatioTarget(
                concurrency=concurrency,
                prefill_instances=prefill_instances,
                decode_instances=decode_instances,
                total_chips=total_chips,
            ),
            cache,
        )
        if point is not None:
            points.append(point)
    return points


def _evaluate_disaggregated_candidates(
    eval_context: _SolveContext,
    decode_states: list[_DisaggDecodeState],
    prefill_options: list[_PrefillOption],
) -> list[ParetoPoint]:
    """Evaluate Cartesian disaggregated P/D candidate pairs when requested."""
    if eval_context.mode not in ("auto", "disagg") or not prefill_options:
        return []
    candidates: list[ParetoPoint] = []
    cache: dict[tuple[str, Parallel, float], tuple[float, float]] = {}
    for dec_state in decode_states:
        for concurrency in eval_context.scenario.concurrencies:
            for pref in prefill_options:
                candidates.extend(
                    _search_disagg_instance_ratios(
                        eval_context, dec_state, pref, concurrency, cache
                    )
                )
    return candidates


def _resolve_solve_topologies(
    opts: SolveOptions,
    chip: ChipSpec | None = None,
) -> list[SliceTopology]:
    """Resolve provisioned topologies within max_chips budget, rejecting subhost."""
    if opts.topologies is not None:
        topologies = list(opts.topologies)
    elif chip is not None:
        topologies = list(topologies_for_chip(chip).values())
    else:
        topologies = list(TOPOLOGIES.values())
    if opts.max_chips is not None:
        if topologies:
            min_topo = min(topologies, key=lambda topology: topology.num_chips)
            if opts.max_chips < min_topo.num_chips:
                msg = (
                    f"max_chips={opts.max_chips} is below minimum provisionable TPU "
                    f"topology {min_topo.name!r} ({min_topo.num_chips} chips)"
                )
                raise ValueError(msg)
        return [
            topology_item
            for topology_item in topologies
            if topology_item.num_chips <= opts.max_chips
        ]
    return topologies


def _validate_solve_options(
    opts: SolveOptions,
    *,
    chip: ChipSpec | None = None,
) -> None:
    """Validate SolveOptions numeric, mode, and topology pin invariants."""
    if not math.isfinite(opts.hbm_reserve_gb) or opts.hbm_reserve_gb < 0.0:
        msg = f"hbm_reserve_gb must be finite and >= 0, got {opts.hbm_reserve_gb}"
        raise ValueError(msg)
    if not math.isfinite(opts.collective_latency_s) or opts.collective_latency_s < 0.0:
        msg = (
            "collective_latency_s must be finite and >= 0, "
            f"got {opts.collective_latency_s}"
        )
        raise ValueError(msg)
    if opts.max_chips is not None and opts.max_chips < 1:
        msg = f"max_chips must be >= 1, got {opts.max_chips}"
        raise ValueError(msg)
    if opts.max_instances is not None and opts.max_instances < 1:
        msg = f"max_instances must be >= 1, got {opts.max_instances}"
        raise ValueError(msg)
    if opts.top_k is not None and opts.top_k < 1:
        msg = f"top_k must be >= 1, got {opts.top_k}"
        raise ValueError(msg)
    if opts.mode == "mixed" and (
        opts.prefill_topology is not None or opts.pd_ratio is not None
    ):
        msg = "mode='mixed' cannot be combined with prefill_topology or pd_ratio"
        raise ValueError(msg)
    if opts.prefill_topology is not None or opts.decode_topology is not None:
        known_topos = (
            {topology_item.name for topology_item in opts.topologies}
            if opts.topologies is not None
            else (
                set(topologies_for_chip(chip)) if chip is not None else set(TOPOLOGIES)
            )
        )
        for pin_name, pin_val in (
            ("prefill_topology", opts.prefill_topology),
            ("decode_topology", opts.decode_topology),
        ):
            if pin_val is not None and pin_val not in known_topos:
                msg = f"Unknown {pin_name} {pin_val!r}; known: {sorted(known_topos)}"
                raise ValueError(msg)


def _active_mode_min_ttft(
    mode: str,
    min_prefill_ttft_ms: float,
    min_mixed_ttft_ms: float,
) -> float:
    """Return minimum single-request TTFT floor for the requested serving mode."""
    if mode == "disagg":
        return min_prefill_ttft_ms
    if mode == "mixed":
        return min_mixed_ttft_ms
    return min(min_prefill_ttft_ms, min_mixed_ttft_ms)


def _collect_warnings(
    geometry: ModelGeometry,
    scenario: TrafficScenario,
    min_floors_ms: tuple[float, float],
    ordered_candidates: tuple[ParetoPoint, ...],
    *,
    has_any_pool: bool = True,
    pools_meet_slo: bool = True,
    disagg_pairs_skipped: bool = False,
) -> tuple[str, ...]:
    """Collect scenario context, capacity, and SLO infeasibility warnings."""
    min_c1_tpot_ms, min_req_ttft_ms = min_floors_ms
    ctx_full = scenario.isl + scenario.osl
    warnings: list[str] = []
    if (
        geometry.max_position_embeddings > 0
        and ctx_full > geometry.max_position_embeddings
    ):
        warnings.append(
            f"CONTEXT_EXCEEDS_MAX_POSITION_EMBEDDINGS: ctx_full={ctx_full} > "
            f"max_position_embeddings={geometry.max_position_embeddings}"
        )
    if (
        scenario.tpot_slo_ms is not None
        and math.isfinite(min_c1_tpot_ms)
        and scenario.tpot_slo_ms < min_c1_tpot_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: tpot_slo_ms={scenario.tpot_slo_ms:.2f} < "
            f"single-request floor {min_c1_tpot_ms:.2f} ms",
        )
    if (
        scenario.ttft_slo_ms is not None
        and math.isfinite(min_req_ttft_ms)
        and scenario.ttft_slo_ms < min_req_ttft_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: ttft_slo_ms={scenario.ttft_slo_ms:.2f} < "
            f"single-request prefill floor {min_req_ttft_ms:.2f} ms",
        )
    if scenario.think_time_s is None and (
        scenario.ttft_slo_ms is not None or scenario.tpot_slo_ms is not None
    ):
        warnings.append(UNKNOWN_THINK_TIME_WARNING)
    if not ordered_candidates and not has_any_pool:
        warnings.append(NO_FEASIBLE_LAYOUT_WARNING)
    elif ordered_candidates and not any(point.slo_met for point in ordered_candidates):
        warnings.append("SLO_UNMET: no feasible topology met both TTFT and TPOT bounds")
    elif not ordered_candidates and has_any_pool and not pools_meet_slo:
        warnings.append("SLO_UNMET: no feasible pool met the active SLO bounds")
    if disagg_pairs_skipped and has_any_pool:
        warnings.append(
            "DISAGG_PAIRS_NOT_EVALUATED: disaggregated pool pairs were not evaluated "
            "(pass include_pairs=True or --pairs)"
        )
    return tuple(warnings)


def _build_pins(opts: SolveOptions) -> dict[str, object] | None:
    """Validate and materialize optional prefill/decode topology and P:D pins."""
    pins: dict[str, object] = {}
    if opts.prefill_topology is not None:
        pins["prefill_topology"] = opts.prefill_topology
    if opts.decode_topology is not None:
        pins["decode_topology"] = opts.decode_topology
    if opts.pd_ratio is not None:
        prefill_instances, decode_instances = opts.pd_ratio
        if prefill_instances < 1 or decode_instances < 1:
            msg = f"pd_ratio must contain positive ints, got {opts.pd_ratio}"
            raise ValueError(msg)
        pins["pd_ratio"] = f"{prefill_instances}:{decode_instances}"
    return pins or None


def _resolve_budget_max_chips(
    opts: SolveOptions,
    *,
    chip: ChipSpec,
) -> int | None:
    """Resolve effective max_chips from explicit budget and complete disagg pins."""
    if (
        opts.pd_ratio is None
        or opts.prefill_topology is None
        or opts.decode_topology is None
    ):
        return opts.max_chips
    topo_map = (
        {topology_item.name: topology_item for topology_item in opts.topologies}
        if opts.topologies is not None
        else topologies_for_chip(chip)
    )
    if opts.prefill_topology not in topo_map or opts.decode_topology not in topo_map:
        return opts.max_chips
    prefill_instances, decode_instances = opts.pd_ratio
    pinned_chips = (
        prefill_instances * topo_map[opts.prefill_topology].num_chips
        + decode_instances * topo_map[opts.decode_topology].num_chips
    )
    if opts.max_chips is not None and opts.max_chips != pinned_chips:
        msg = (
            f"max_chips={opts.max_chips} conflicts with pinned "
            f"{prefill_instances}x{opts.prefill_topology} + "
            f"{decode_instances}x{opts.decode_topology} "
            f"({pinned_chips} chips)"
        )
        raise ValueError(msg)
    return pinned_chips


def _count_disagg_enumerated(
    eval_context: _SolveContext,
    topologies: Sequence[SliceTopology],
    *,
    include_pairs: bool,
) -> int:
    """Count legal disaggregated (layout, concurrency) tuples before HBM filtering."""
    if eval_context.mode not in ("auto", "disagg"):
        return 0
    conc_count = len(eval_context.scenario.concurrencies)
    if include_pairs:
        total = 0
        for d_topo in topologies:
            if (
                eval_context.decode_topology is not None
                and d_topo.name != eval_context.decode_topology
            ):
                continue
            d_count = len(_candidate_decode_parallels(eval_context.geometry, d_topo))
            for p_topo in topologies:
                if (
                    eval_context.prefill_topology is not None
                    and p_topo.name != eval_context.prefill_topology
                ):
                    continue
                p_count = len(
                    _candidate_prefill_parallels(eval_context.geometry, p_topo)
                )
                slots = _count_disagg_ratio_slots(eval_context, p_topo, d_topo)
                total += p_count * d_count * slots * conc_count
        return total
    if eval_context.mode != "disagg":
        return 0
    pool_total = 0
    for topology in topologies:
        if (
            eval_context.prefill_topology is None
            or topology.name == eval_context.prefill_topology
        ):
            pool_total += (
                len(_candidate_prefill_parallels(eval_context.geometry, topology))
                * conc_count
            )
        if (
            eval_context.decode_topology is None
            or topology.name == eval_context.decode_topology
        ):
            pool_total += (
                len(_candidate_decode_parallels(eval_context.geometry, topology))
                * conc_count
            )
    return pool_total


def solve_serving_plan(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    options: SolveOptions | None = None,
) -> ServingPlan:
    """Evaluate all physics-legal layouts, mark Pareto members, and order per pool."""
    opts = options or SolveOptions()
    _validate_solve_options(opts, chip=chip)
    pins = _build_pins(opts)
    eff_max_chips = _resolve_budget_max_chips(opts, chip=chip)
    eff_opts = (
        replace(opts, max_chips=eff_max_chips)
        if eff_max_chips != opts.max_chips
        else opts
    )
    topos = _resolve_solve_topologies(eff_opts, chip=chip)
    hbm_limit_b = max(0.0, float(chip.hbm_bytes) - eff_opts.hbm_reserve_gb * UNIT_GIGA)
    effective_accept_len = max(1.0, scenario.accept_len)
    spec_k = _resolve_spec_k(geometry, scenario)
    ctx_full = scenario.isl + scenario.osl
    eff_isl = max(1, math.ceil((1.0 - scenario.prefix_hit_rate) * scenario.isl))
    n_chunks = max(1, math.ceil(eff_isl / max(1, scenario.prefill_chunk)))
    should_eval_pairs = eff_opts.include_pairs or eff_opts.pd_ratio is not None

    eval_context = _SolveContext(
        geometry=geometry,
        dtypes=dtypes,
        chip=chip,
        scenario=scenario,
        mode=eff_opts.mode,
        max_chips=eff_opts.max_chips,
        max_instances=eff_opts.max_instances,
        hbm_reserve_gb=eff_opts.hbm_reserve_gb,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / UNIT_GIGA,
        effective_accept_len=effective_accept_len,
        spec_k=spec_k,
        ctx_full=ctx_full,
        ctx_mid=scenario.isl + scenario.osl // 2,
        eff_isl=eff_isl,
        n_chunks=n_chunks,
        c_max=max(scenario.concurrencies),
        shared_prefix_tokens=_shared_prefix_tokens(scenario),
        collective_latency_s=eff_opts.collective_latency_s,
        sort_by=eff_opts.sort_by,
        prefill_topology=eff_opts.prefill_topology,
        decode_topology=eff_opts.decode_topology,
        pd_ratio=eff_opts.pd_ratio,
        activation_model=eff_opts.activation_model,
    )
    core_grid = _count_core_grid(eval_context, topos, include_pairs=should_eval_pairs)
    eval_context.enumerated += _count_disagg_enumerated(
        eval_context, topos, include_pairs=should_eval_pairs
    )

    prefill_options, min_prefill_ttft_ms = (
        _find_prefill_options(eval_context, topos)
        if eff_opts.mode in ("auto", "disagg")
        else ([], math.inf)
    )
    mixed_candidates, min_c1_tpot_ms, min_mixed_ttft_ms = _evaluate_mixed_candidates(
        eval_context, topos
    )
    decode_states = (
        _build_decode_pool_states(eval_context, topos)
        if eff_opts.mode in ("auto", "disagg")
        else []
    )
    prefill_pools = _order_and_mark_pool_records(
        _build_prefill_pool_records(eval_context, prefill_options),
        eff_opts.sort_by,
    )
    decode_pools = _order_and_mark_pool_records(
        _build_decode_pool_records(eval_context, decode_states),
        eff_opts.sort_by,
    )
    disagg_candidates = (
        _evaluate_disaggregated_candidates(eval_context, decode_states, prefill_options)
        if should_eval_pairs
        else []
    )
    all_candidates = mixed_candidates + disagg_candidates
    ordered_candidates, frontier = _order_and_mark_candidates(
        all_candidates,
        eff_opts.sort_by,
    )
    pools_meet_slo = (
        not prefill_pools or any(record.slo_met for record in prefill_pools)
    ) and (not decode_pools or any(record.slo_met for record in decode_pools))
    warnings = _collect_warnings(
        geometry,
        scenario,
        (
            min_c1_tpot_ms,
            _active_mode_min_ttft(
                eff_opts.mode, min_prefill_ttft_ms, min_mixed_ttft_ms
            ),
        ),
        ordered_candidates,
        has_any_pool=bool(prefill_pools or decode_pools),
        pools_meet_slo=pools_meet_slo,
        disagg_pairs_skipped=(
            eff_opts.mode in ("auto", "disagg") and not should_eval_pairs
        ),
    )
    meets_cands = [
        point
        for point in ordered_candidates
        if point.slo_verdict == SLO_VERDICT_ESTIMATE_MEETS
    ]
    not_ruled_cands = [
        point
        for point in ordered_candidates
        if point.slo_verdict != SLO_VERDICT_RULED_OUT
    ]
    rec_pool = meets_cands or not_ruled_cands
    recommended = (
        min(
            rec_pool,
            key=lambda point: _candidate_sort_key(point, eff_opts.sort_by),
        )
        if rec_pool
        else None
    )
    warn_list = list(warnings)
    if recommended is not None and recommended.slo_verdict == SLO_VERDICT_UNDECIDED:
        warn_list.append(UNDECIDED_SLO_WARNING)
    warnings = tuple(warn_list)
    not_ruled_out_count = sum(
        1 for point in all_candidates if point.slo_verdict != SLO_VERDICT_RULED_OUT
    )
    env_feasible_count = sum(
        1
        for point in all_candidates
        if point.slo_verdict != SLO_VERDICT_RULED_OUT
        and _within_slo_bounds(scenario, point.envelope_ttft_ms, point.envelope_tpot_ms)
    )
    search_stats = SearchStats(
        enumerated=eval_context.enumerated,
        hbm_fit=eval_context.hbm_fit,
        slo_met=sum(1 for point in all_candidates if point.slo_met),
        frontier=len(frontier),
        core_grid=max(core_grid, eval_context.enumerated),
        not_ruled_out=not_ruled_out_count,
        envelope_feasible=env_feasible_count,
    )
    budget_val = (
        eff_max_chips
        if eff_max_chips is not None
        else max((topology_item.num_chips for topology_item in topos), default=0)
    )
    budget_source = (
        "flag"
        if opts.max_chips is not None
        else ("pins" if eff_max_chips is not None else "max_instances")
    )
    return ServingPlan(
        model_layers=geometry.num_layers,
        chip=chip.name,
        scenario=scenario,
        mode=eff_opts.mode,
        recommended=recommended,
        frontier=frontier,
        warnings=warnings,
        candidates=ordered_candidates,
        prefill_pools=prefill_pools,
        decode_pools=decode_pools,
        hbm_reserve_gb=eff_opts.hbm_reserve_gb,
        collective_latency_s=eff_opts.collective_latency_s,
        spec_k=spec_k,
        spec_tokens=scenario.spec_tokens or 0,
        mtp_layers=geometry.mtp_layers,
        sort_by=eff_opts.sort_by,
        model_id=geometry.model_id,
        search=search_stats,
        has_moe=geometry.num_experts > 0,
        budget=budget_val,
        budget_source=budget_source,
        max_instances=eff_opts.max_instances,
        pins=pins,
        activation_model=eff_opts.activation_model,
        ruled_out_candidates=tuple(
            point
            for point in all_candidates
            if point.slo_verdict == SLO_VERDICT_RULED_OUT
        ),
    )


def _explain_mode_rejection(
    layout: Mapping[str, Mapping[str, int]],
    mode: str,
) -> str | None:
    """Return rejection reason if layout contradicts solver mode."""
    if mode == "mixed" and "prefill" in layout:
        return "a prefill and decode layout is only searched in disaggregated mode"
    if mode == "disagg" and "prefill" not in layout:
        return "a single-pool layout is only searched in mixed mode"
    return None


def _explain_pool_axis_rejection(
    geometry: ModelGeometry,
    axes: Mapping[str, int],
    role_label: str,
) -> str | None:
    """Return first physical divisibility rejection for a single pool layout."""
    tp = axes.get("tp", 1)
    dp = axes.get("dp", 1)
    ep = axes.get("ep", 1)
    pp = axes.get("pp", 1)
    pcp = axes.get("pcp", 1)
    kv_split = axes.get("kv_split", 1)
    if not _valid_head_tp(geometry, tp):
        return f"{role_label} TP{tp} does not divide the attention heads"
    if not _valid_gdn_degree(geometry, tp):
        return f"{role_label} TP{tp} is not a valid linear-attention shard degree"
    if pp > geometry.num_layers:
        return f"{role_label} PP{pp} exceeds model layers ({geometry.num_layers})"
    if kv_split not in _kv_splits_for_tp(geometry, tp):
        return f"{role_label} DCP{kv_split} is not physically valid for TP{tp}"
    ep_domain = tp * dp * pcp
    legal_moe = _candidate_pool_moe_modes(geometry, tp, ep_domain)
    if not any(ep_val == ep for _, ep_val in legal_moe):
        return (
            f"{role_label} EP{ep} is not physically divisible for "
            f"TP{tp} x DP{dp} x PCP{pcp}"
        )
    return None


def _explain_pool_rejection(
    geometry: ModelGeometry,
    axes: Mapping[str, int],
    topologies: Sequence[SliceTopology],
    role_label: str,
    *,
    is_disagg: bool = False,
    pinned_topology: str | None = None,
) -> str | None:
    """Return first physical or slice-size rejection for a pool layout."""
    axis_reason = _explain_pool_axis_rejection(geometry, axes, role_label)
    if axis_reason is not None:
        return axis_reason
    tp = axes.get("tp", 1)
    dp = axes.get("dp", 1)
    pp = axes.get("pp", 1)
    pcp = axes.get("pcp", 1)
    ep = axes.get("ep", 1)
    kv_split = axes.get("kv_split", 1)
    if (not is_disagg or role_label == "decode") and pcp > 1:
        return (
            "decode pool requires PCP1 in disaggregated mode"
            if is_disagg
            else "mixed layout requires PCP1"
        )
    needed_cores = tp * dp * pp * pcp
    if pinned_topology is not None and not topologies:
        return (
            f"{role_label} topology pin {pinned_topology!r} is not in the "
            "searched topologies"
        )
    matching_topos = [
        topology for topology in topologies if topology.num_cores == needed_cores
    ]
    if not matching_topos:
        if pinned_topology is not None:
            pinned_cores = topologies[0].num_cores
            return (
                f"{role_label} layout needs {needed_cores} cores, which does not "
                f"match pinned {role_label} topology {pinned_topology} "
                f"({pinned_cores} cores)"
            )
        return (
            f"{role_label} layout needs {needed_cores} cores and no searched "
            "slice has that many"
        )
    if not is_disagg or role_label == "decode":
        enum_fn = _candidate_decode_parallels
    else:
        enum_fn = _candidate_prefill_parallels
    found = any(
        cand.tp == tp
        and cand.dp == dp
        and cand.pp == pp
        and cand.pcp == pcp
        and cand.ep == ep
        and cand.kv_split == kv_split
        for topology in matching_topos
        for cand in enum_fn(geometry, topology)
    )
    if not found:
        return (
            f"this {role_label} layout is not enumerated on a {needed_cores}-core slice"
        )
    return None


def _min_matching_chips(
    axes: Mapping[str, int],
    topologies: Sequence[SliceTopology],
) -> int | None:
    """Return smallest chip count among topologies matching a pool's core count."""
    needed_cores = (
        axes.get("tp", 1) * axes.get("dp", 1) * axes.get("pp", 1) * axes.get("pcp", 1)
    )
    chips = [
        topology.num_chips
        for topology in topologies
        if topology.num_cores == needed_cores
    ]
    return min(chips) if chips else None


def explain_layout_rejection(
    geometry: ModelGeometry,
    layout: Mapping[str, Mapping[str, int]],
    topologies: Sequence[SliceTopology],
    opts: SolveOptions | None = None,
) -> str:
    """Name the first physical rule that leaves a reference layout absent."""
    options = opts or SolveOptions()
    mode_reason = _explain_mode_rejection(layout, options.mode)
    if mode_reason is not None:
        return mode_reason
    is_disagg = "prefill" in layout
    decode_topologies = [
        topology
        for topology in topologies
        if options.decode_topology is None or topology.name == options.decode_topology
    ]
    prefill_topologies = [
        topology
        for topology in topologies
        if options.prefill_topology is None or topology.name == options.prefill_topology
    ]
    decode_reason = _explain_pool_rejection(
        geometry,
        layout["decode"],
        decode_topologies,
        "decode",
        is_disagg=is_disagg,
        pinned_topology=options.decode_topology,
    )
    if decode_reason is not None:
        return decode_reason
    min_decode_chips = _min_matching_chips(layout["decode"], decode_topologies)
    if is_disagg:
        prefill_reason = _explain_pool_rejection(
            geometry,
            layout["prefill"],
            prefill_topologies,
            "prefill",
            is_disagg=True,
            pinned_topology=options.prefill_topology,
        )
        if prefill_reason is not None:
            return prefill_reason
        if options.max_instances is not None:
            if options.pd_ratio is not None:
                prefill_inst, decode_inst = options.pd_ratio
                if prefill_inst + decode_inst > options.max_instances:
                    return (
                        f"disaggregated layout with --pd {prefill_inst}:{decode_inst} "
                        f"needs {prefill_inst + decode_inst} instances, exceeding "
                        f"max_instances={options.max_instances}"
                    )
            elif options.max_instances < MIN_DISAGG_INSTANCES:
                return (
                    "disaggregated layout needs at least "
                    f"{MIN_DISAGG_INSTANCES} instances (1P+1D), exceeding "
                    f"max_instances={options.max_instances}"
                )
        min_prefill_chips = _min_matching_chips(layout["prefill"], prefill_topologies)
        if (
            options.max_chips is not None
            and min_decode_chips is not None
            and min_prefill_chips is not None
        ):
            if options.pd_ratio is not None:
                prefill_inst, decode_inst = options.pd_ratio
                needed_chips = (
                    prefill_inst * min_prefill_chips + decode_inst * min_decode_chips
                )
                if needed_chips > options.max_chips:
                    return (
                        f"disaggregated layout with --pd {prefill_inst}:{decode_inst} "
                        f"needs at least {needed_chips} chips, exceeding "
                        f"max_chips={options.max_chips}"
                    )
            elif min_prefill_chips + min_decode_chips > options.max_chips:
                return (
                    "disaggregated layout needs at least "
                    f"{min_prefill_chips + min_decode_chips} chips (1P+1D), "
                    f"exceeding max_chips={options.max_chips}"
                )
        if not options.include_pairs and options.pd_ratio is None:
            return (
                "disaggregated pair candidates are off by default; "
                "pass --pairs or --pd to evaluate pairs"
            )
    elif (
        options.max_chips is not None
        and min_decode_chips is not None
        and min_decode_chips > options.max_chips
    ):
        return (
            f"decode layout needs at least {min_decode_chips} chips, "
            f"exceeding max_chips={options.max_chips}"
        )
    return (
        "enumerated, but no searched point with this layout passed the "
        "memory fit, the chip budget, and the latency targets"
    )
