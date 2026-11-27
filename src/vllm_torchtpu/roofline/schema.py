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
"""Shared serving-plan dataclasses, payload TypedDicts, and formatting helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from .chips import count_hosts
from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_HBM_RESERVE_GB,
    DEFAULT_PREFILL_CHUNK,
    SLO_VERDICT_ESTIMATE_MEETS,
    SLO_VERDICT_RANK,
    SLO_VERDICT_RULED_OUT,
    TOPOLOGY_AXES,
    SloVerdict,
)
from .occupancy import ExpertActivationModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chips import ChipSpec, SliceTopology
    from .geometry import ExecutionMode, HostTableResidency, ModelGeometry

type SortBy = Literal["throughput", "interactivity"]

_POOL_AXES: tuple[tuple[str, str], ...] = (
    ("TP", "tp"),
    ("DP", "dp"),
    ("EP", "ep"),
    ("PP", "pp"),
    ("PCP", "pcp"),
    ("DCP", "kv_split"),
)
_ATTN_DISPLAY_AXES: tuple[tuple[str, str], ...] = tuple(
    (label, key) for label, key in _POOL_AXES if label not in ("EP", "PP")
)


@dataclass(frozen=True, slots=True)
class TrafficScenario:
    """User-selected workload distribution and optional latency SLO bounds."""

    name: str
    isl: int
    osl: int
    prefix_hit_rate: float
    accept_len: float
    think_time_s: float | None
    concurrencies: tuple[int, ...]
    kv_working_set_turns: float
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    spec_tokens: int | None = None
    kv_offload_gb: float = 0.0
    execution: ExecutionMode = "full"
    host_table_residency: HostTableResidency = "auto"
    prefill_draft: bool = False

    def __post_init__(self) -> None:
        """Validate input sequence lengths, hit rates, and concurrency parameters."""
        if self.isl <= 0 or self.osl <= 0:
            msg = f"isl and osl must be > 0, got {self.isl}, {self.osl}"
            raise ValueError(msg)
        if not math.isfinite(self.prefix_hit_rate) or not (
            0.0 <= self.prefix_hit_rate <= 1.0
        ):
            msg = f"prefix_hit_rate must be in [0, 1], got {self.prefix_hit_rate}"
            raise ValueError(msg)
        if not math.isfinite(self.accept_len) or self.accept_len < 1.0:
            msg = f"accept_len must be >= 1.0, got {self.accept_len}"
            raise ValueError(msg)
        self._validate_spec_tokens()
        if self.think_time_s is not None and (
            not math.isfinite(self.think_time_s) or self.think_time_s < 0.0
        ):
            msg = f"think_time_s must be >= 0, got {self.think_time_s}"
            raise ValueError(msg)
        if (
            not math.isfinite(self.kv_working_set_turns)
            or self.kv_working_set_turns <= 0.0
        ):
            msg = (
                "kv_working_set_turns must be finite and > 0, "
                f"got {self.kv_working_set_turns}"
            )
            raise ValueError(msg)
        if self.prefill_chunk <= 0:
            msg = f"prefill_chunk must be > 0, got {self.prefill_chunk}"
            raise ValueError(msg)
        if self.ttft_slo_ms is not None and (
            not math.isfinite(self.ttft_slo_ms) or self.ttft_slo_ms <= 0.0
        ):
            msg = f"ttft_slo_ms must be finite and > 0, got {self.ttft_slo_ms}"
            raise ValueError(msg)
        if self.tpot_slo_ms is not None and (
            not math.isfinite(self.tpot_slo_ms) or self.tpot_slo_ms <= 0.0
        ):
            msg = f"tpot_slo_ms must be finite and > 0, got {self.tpot_slo_ms}"
            raise ValueError(msg)
        if not math.isfinite(self.kv_offload_gb) or self.kv_offload_gb < 0.0:
            msg = f"kv_offload_gb must be finite and >= 0, got {self.kv_offload_gb}"
            raise ValueError(msg)
        if not self.concurrencies or any(
            concurrency <= 0 for concurrency in self.concurrencies
        ):
            msg = f"concurrencies must be positive ints, got {self.concurrencies}"
            raise ValueError(msg)
        self._validate_physical_modes()

    def _validate_physical_modes(self) -> None:
        """Validate optional execution and host-table residency modes."""
        if self.execution not in ("full", "ced"):
            msg = f"execution must be 'full' or 'ced', got {self.execution!r}"
            raise ValueError(msg)
        if self.host_table_residency not in ("auto", "hbm", "host", "first"):
            msg = (
                "host_table_residency must be one of auto/hbm/host/first, "
                f"got {self.host_table_residency!r}"
            )
            raise ValueError(msg)

    def _validate_spec_tokens(self) -> None:
        """Validate speculative decoding token count against accept_len."""
        if self.spec_tokens is None:
            return
        if self.spec_tokens < 0:
            msg = f"spec_tokens must be >= 0, got {self.spec_tokens}"
            raise ValueError(msg)
        if self.accept_len > 1.0 + float(self.spec_tokens):
            msg = (
                f"accept_len={self.accept_len} exceeds "
                f"1 + spec_tokens={1 + self.spec_tokens}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ScenarioOverrides:
    """Optional field overrides applied when building an example TrafficScenario."""

    isl: int | None = None
    osl: int | None = None
    concurrencies: tuple[int, ...] | None = None
    prefix_hit_rate: float | None = None
    accept_len: float | None = None
    think_time_s: float | None = None
    prefill_chunk: int | None = None
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    kv_working_set_turns: float | None = None
    kv_offload_gb: float | None = None
    execution: ExecutionMode | None = None
    host_table_residency: HostTableResidency | None = None
    prefill_draft: bool | None = None


@dataclass(frozen=True, slots=True)
class FloorTimes:
    """Decode-step resource times (ms) behind a point's binding unit."""

    hbm_ms: float
    matmul_ms: float
    ici_ms: float
    chain_ms: float
    hop_latency_ms: float
    hard_floor_ms: float
    binding: str
    hbm_bound_ms: float = 0.0
    matmul_bound_ms: float = 0.0
    vector_bound_ms: float = 0.0
    coll_bound_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Search funnel: (layout, concurrency) tuples inside the chip budget."""

    enumerated: int
    hbm_fit: int
    slo_met: int
    frontier: int
    core_grid: int = 0
    not_ruled_out: int = 0
    envelope_feasible: int = 0


@dataclass(frozen=True, slots=True)
class PoolMeshPlan:
    """Homogeneous instance pool for mixed, prefill-only, or decode-only role."""

    role: Literal["mixed", "prefill", "decode"]
    instances: int
    topology: str
    chips_per_instance: int
    tp: int
    dp: int
    ep: int
    pp: int
    pcp: int
    kv_split: int
    mesh_axes: tuple[int, int, int, int]
    hbm_used_gb: float
    hbm_limit_gb: float
    moe_dispatch: Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"] = "allreduce"
    hbm_weights_gb: float = 0.0
    hbm_kv_gb: float = 0.0
    hbm_state_gb: float = 0.0
    resident_batch_cap: int | None = None
    residual: Literal["replicated", "sp"] = "replicated"
    pp_layer_partition: tuple[int, ...] | None = None
    kv_offload_gb: float = 0.0


@dataclass(frozen=True, slots=True)
class PoolFloorRecord:
    """Per-concurrency physics evaluation for one prefill or decode pool layout."""

    role: Literal["prefill", "decode"]
    concurrency: int
    pool: PoolMeshPlan
    ttft_ms: float | None
    ttft_floor_ms: float | None
    tpot_ms: float | None
    tpot_floor_ms: float | None
    interactivity_tok_s: float | None
    token_throughput_per_chip: float
    request_rate_per_instance: float
    binding: str
    slo_met: bool
    is_pareto: bool = False
    kv_cache_capacity_tokens: int | None = None
    effective_prefix_hit_rate: float | None = None


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """Evaluated serving configuration with latency, throughput, and SLO state."""

    mode: Literal["mixed", "disagg"]
    concurrency: int
    total_chips: int
    prefill_pool: PoolMeshPlan | None
    decode_pool: PoolMeshPlan
    est_ttft_ms: float
    est_tpot_ms: float
    est_interactivity_tok_s: float
    output_throughput_per_chip: float
    token_throughput_per_chip: float
    p_mixed: float
    binding: str
    slo_met: bool
    bound_tpot_ms: float = 0.0
    bound_ttft_ms: float = 0.0
    slo_verdict: SloVerdict = SLO_VERDICT_ESTIMATE_MEETS
    decode_floor: FloorTimes | None = None
    kv_cache_capacity_tokens: int | None = None
    effective_prefix_hit_rate: float | None = None
    min_decode_seqs: float = 0.0
    floor_throughput_per_chip: float = 0.0
    envelope_tpot_ms: float = 0.0
    envelope_ttft_ms: float = 0.0
    envelope_decode_seqs: float = 0.0
    is_pareto: bool = False


@dataclass(frozen=True, slots=True)
class ServingPlan:
    """Ordered physics-legal layouts, Pareto marks, and per-pool floor tables."""

    model_layers: int
    chip: str
    scenario: TrafficScenario
    mode: str
    recommended: ParetoPoint | None
    frontier: tuple[ParetoPoint, ...]
    warnings: tuple[str, ...]
    candidates: tuple[ParetoPoint, ...] = ()
    prefill_pools: tuple[PoolFloorRecord, ...] = ()
    decode_pools: tuple[PoolFloorRecord, ...] = ()
    hbm_reserve_gb: float = DEFAULT_HBM_RESERVE_GB
    collective_latency_s: float = 0.0
    spec_k: int = 0
    spec_tokens: int = 0
    mtp_layers: int = 0
    sort_by: SortBy = "throughput"
    model_id: str = ""
    search: SearchStats | None = None
    has_moe: bool = False
    budget: int = 0
    budget_source: str = "flag"
    max_instances: int | None = None
    pins: dict[str, object] | None = None
    activation_model: ExpertActivationModel = field(
        default_factory=ExpertActivationModel
    )
    ruled_out_candidates: tuple[ParetoPoint, ...] = ()
    sweep_candidates: tuple[ParetoPoint, ...] = ()
    winner_prefix_hit_range: tuple[float, float] | None = None
    prefix_hit_crossover_rate: float | None = None
    draft_config_source: str | None = None


class ProvenancePayload(TypedDict):
    """Analytical floor and uncalibrated estimate provenance in solve JSON."""

    calibrated: bool
    basis: str
    supplied_values: NotRequired[list[dict[str, object]]]
    ranking_basis: NotRequired[str]


class TopKEntry(TypedDict):
    """Ranked candidate entry with role label, margin, and metrics."""

    frontier_role: str
    margin_vs_rank1_pct: float
    point: Mapping[str, object]


class BoundFamilyEntry(TypedDict):
    """Bound-ordered structural layout family summary at one concurrency."""

    rank: int
    family: tuple[str, ...]
    min_bound_tpot_ms: float | None
    min_bound_ttft_ms: float | None
    binding: str
    any_not_ruled_out: bool
    member_count: int
    members: list[dict[str, object]]
    min_envelope_tpot_ms: NotRequired[float | None]
    min_envelope_ttft_ms: NotRequired[float | None]


class ReferenceLayoutReport(TypedDict):
    """Ranking and diagnostic report for a requested reference layout."""

    layout: str
    found: bool
    ruled_out: bool
    concurrency: int | None
    candidates: int
    rank_by_tpot_bound: int | None
    rank_by_estimated_throughput: int | None
    in_top_k: bool
    point: Mapping[str, object] | None
    family_in_ordered_top_k: bool
    ordered_family_peer: str | None
    rejected_by: str
    rejection_reason: str
    slo_verdict: SloVerdict | None
    bound_tpot_ms: float | None
    bound_ttft_ms: float | None
    floor_throughput_per_chip: float | None
    hbm_used_gb: float | None
    min_decode_seqs: float | None


class SolvePayload(TypedDict):
    """Versioned solve JSON payload with provenance and ordered per-pool lists."""

    model_layers: int
    chip: str
    scenario: Mapping[str, object]
    mode: str
    recommended: Mapping[str, object] | None
    frontier: tuple[Mapping[str, object], ...]
    warnings: tuple[str, ...]
    candidates: tuple[Mapping[str, object], ...]
    prefill_pools: tuple[Mapping[str, object], ...]
    decode_pools: tuple[Mapping[str, object], ...]
    hbm_reserve_gb: float
    sort_by: SortBy
    top_k_limit: int | None
    search: Mapping[str, object] | None
    schema: str
    provenance: ProvenancePayload
    chip_spec: Mapping[str, object]
    model: str | None
    top_k: list[TopKEntry]
    has_moe: bool
    single_host_fit: dict[str, object]
    target_chips: int
    budget: int
    budget_source: str
    max_instances: int | None
    pins: Mapping[str, object] | None
    chip_budget: Mapping[str, object]
    dtypes: NotRequired[Mapping[str, object]]
    concurrency_top_k: NotRequired[dict[str, list[TopKEntry]]]
    bound_families_by_concurrency: NotRequired[dict[str, list[BoundFamilyEntry]]]
    solver_assumptions: NotRequired[list[dict[str, object]]]
    winner_prefix_hit_range: NotRequired[tuple[float, float] | None]
    prefix_hit_crossover_rate: NotRequired[float | None]
    reference: NotRequired[ReferenceLayoutReport | Mapping[str, object]]
    activation_regime: NotRequired[str]
    spec_tokens: NotRequired[int]
    mtp_layers: NotRequired[int]


class CeilingRow(TypedDict):
    """Single physics candidate row exported by ``emit_ceiling_rows``."""

    est_interactivity_tok_s: float
    token_throughput_per_chip: float
    output_throughput_per_chip: float
    est_ttft_ms: float
    est_tpot_ms: float
    bound_tpot_ms: float
    concurrency: int
    total_chips: int
    mode: str
    binding_unit: str
    min_decode_seqs: float
    floor_throughput_per_chip: float
    slo_met: bool
    is_pareto: bool
    decode_topology: str
    decode_parallelism: str
    prefill_topology: str | None
    prefill_parallelism: str | None


type PoolStructKey = tuple[
    str, int, int, int, int, int, int, int, tuple[int, ...] | None
]


def effective_slo_verdict(point: ParetoPoint) -> SloVerdict:
    """Resolve explicit SLO verdict or derive binary fallback from ``slo_met``."""
    if point.slo_verdict is not None:
        return point.slo_verdict
    return SLO_VERDICT_ESTIMATE_MEETS if point.slo_met else SLO_VERDICT_RULED_OUT


def _pool_int(val: object, default: int = 1) -> int:
    """Return integer pool attribute value when present, else default."""
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    return default


def format_parallelism(pool: Mapping[str, object]) -> str:
    """Format active parallelism axes joined by ' x ' in canonical order, or TP1."""
    parts = [
        f"{name}{_pool_int(pool.get(key))}"
        for name, key in _POOL_AXES
        if _pool_int(pool.get(key)) > 1
    ]
    return " x ".join(parts) if parts else "TP1"


def format_host_count(hosts: int) -> str:
    """Format host count with singular or plural noun."""
    return "1 host" if hosts == 1 else f"{hosts} hosts"


def _pool_slice_counts(
    pool: Mapping[str, object],
    *,
    topologies: Mapping[str, SliceTopology] | None = None,
) -> tuple[int, str, int]:
    """Return (instance_count, topology_name, total_chips) for a pool mapping."""
    instance_count = _pool_int(pool.get("instances"))
    topology_name = str(pool.get("topology", ""))
    chips_per_instance = _pool_int(pool.get("chips_per_instance"), default=0)
    if chips_per_instance <= 0:
        if topologies is not None and topology_name in topologies:
            chips_per_instance = topologies[topology_name].num_chips
        elif topology_name in TOPOLOGY_AXES:
            dim_x, dim_y, dim_z = TOPOLOGY_AXES[topology_name]
            chips_per_instance = dim_x * dim_y * dim_z
    total_chips = chips_per_instance * max(1, instance_count)
    return instance_count, topology_name, total_chips


def _format_slice_hw(total_chips: int, chip: ChipSpec | None) -> tuple[str, int]:
    """Return host description and TensorCore count for a slice chip total."""
    chips_per_host = chip.chips_per_host if chip is not None else DEFAULT_CHIPS_PER_HOST
    cores_per_chip = chip.cores_per_chip if chip is not None else DEFAULT_CORES_PER_CHIP
    host_description = format_host_count(
        count_hosts(total_chips, chips_per_host=chips_per_host)
    )
    return host_description, total_chips * cores_per_chip


def format_submodule_parallelism(
    pool: Mapping[str, object],
    *,
    has_moe: bool = True,
) -> str:
    """Format pool axes with explicit Attention and MoE submodule sharding."""
    ep_degree = _pool_int(pool.get("ep"))
    if not has_moe and ep_degree <= 1:
        return format_parallelism(pool)
    pp_degree = _pool_int(pool.get("pp"))
    tp_degree = _pool_int(pool.get("tp"))
    attn_parts = [
        f"{axis_name}{_pool_int(pool.get(axis_key))}"
        for axis_name, axis_key in _ATTN_DISPLAY_AXES
        if _pool_int(pool.get(axis_key)) > 1
    ]
    attn_text = " x ".join(attn_parts) if attn_parts else "TP1"
    if 1 < ep_degree < tp_degree and tp_degree % ep_degree == 0:
        moe_text = f"EP{ep_degree} x TP{tp_degree // ep_degree}"
    elif ep_degree > 1:
        moe_text = f"EP{ep_degree}"
    else:
        moe_text = f"TP{tp_degree}"
    if pp_degree > 1:
        return f"PP{pp_degree} · Attn: {attn_text} · MoE: {moe_text}"
    return f"Attn: {attn_text} · MoE: {moe_text}"


def pool_slice_summary(
    pool: Mapping[str, object],
    *,
    chip: ChipSpec | None = None,
    topologies: Mapping[str, SliceTopology] | None = None,
) -> str:
    """Format compact instance and slice hardware counts without parallelism axes."""
    instance_count, topology_name, total_chips = _pool_slice_counts(
        pool, topologies=topologies
    )
    if total_chips <= 0:
        return f"{instance_count}x {topology_name}"
    host_description, cores = _format_slice_hw(total_chips, chip)
    return (
        f"{instance_count}x {topology_name} · "
        f"{total_chips} chips ({host_description}, {cores} cores)"
    )


def pool_label(
    pool: Mapping[str, object],
    *,
    chip: ChipSpec | None = None,
    topologies: Mapping[str, SliceTopology] | None = None,
) -> str:
    """Return a concise pool label summarizing topology, hardware, and axes."""
    instance_count, topology_name, total_chips = _pool_slice_counts(
        pool, topologies=topologies
    )
    instance_prefix = f"{instance_count}x "
    layout_text = format_parallelism(pool)
    if total_chips <= 0:
        return f"{instance_prefix}{topology_name}: {layout_text}"
    host_description, cores = _format_slice_hw(total_chips, chip)
    pool_description = f" ({total_chips} chips, {host_description}, {cores} cores)"
    return f"{instance_prefix}{topology_name}{pool_description}: {layout_text}"


def _pool_key(pool: PoolMeshPlan | None) -> PoolStructKey | None:
    """Group layouts by structural topology, instance count, and parallelism axes."""
    if pool is None:
        return None
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.pp_layer_partition,
    )


def _point_struct_key(
    point: ParetoPoint,
) -> tuple[str, PoolStructKey | None, PoolStructKey | None]:
    """Build composite structural identity key across mixed or prefill/decode pools."""
    return (point.mode, _pool_key(point.prefill_pool), _pool_key(point.decode_pool))


def shared_prefix_tokens(scenario: TrafficScenario) -> int:
    """Return rounded shared-prefix token count for one traffic scenario."""
    return round(scenario.prefix_hit_rate * scenario.isl)


def resolve_spec_k(
    geometry: ModelGeometry,
    scenario: TrafficScenario,
    spec_k: int | None = None,
) -> int:
    """Resolve effective speculative draft step count for a scenario."""
    if spec_k is not None:
        return spec_k
    if scenario.spec_tokens is not None:
        return scenario.spec_tokens
    effective_accept_len = max(1.0, scenario.accept_len)
    if effective_accept_len > 1.0:
        default_k = (
            max(geometry.draft.block_size, geometry.draft.num_layers)
            if geometry.draft is not None
            else geometry.mtp_layers
        )
        return max(default_k, math.ceil(effective_accept_len) - 1)
    return 0


def candidate_sort_key(
    point: ParetoPoint,
    sort_by: SortBy,
) -> tuple[int, float, float, float, int, str]:
    """Build unified candidate sort key shared by solver, emit, and families.

    Orders by 3-valued SLO verdict rank, primary estimate metric, secondary
    estimate metric (breaking ties toward Pareto-optimal points), ceiling
    throughput tie-break, total chips, and structural identity.
    """
    verdict_rank = -SLO_VERDICT_RANK[effective_slo_verdict(point)]
    primary, secondary = (
        (point.est_interactivity_tok_s, point.token_throughput_per_chip)
        if sort_by == "interactivity"
        else (point.token_throughput_per_chip, point.est_interactivity_tok_s)
    )
    return (
        verdict_rank,
        -primary,
        -secondary,
        -point.floor_throughput_per_chip,
        point.total_chips,
        str(_point_struct_key(point)),
    )
