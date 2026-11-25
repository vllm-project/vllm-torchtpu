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

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from .chips import TOPOLOGIES, ChipSpec, SliceTopology, topologies_for_chip
from .collectives import submesh_for_group
from .constants import (
    CAUSAL_MIDPOINT_HALF,
    DEFAULT_HBM_RESERVE_GB,
    DEFAULT_MAX_INSTANCES,
    EP_PAIR_RANKS,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    LINEAR_ATTENTION_LAYER_TOKENS,
    NO_FEASIBLE_LAYOUT_WARNING,
    UNIT_GIGA,
    UNIT_MILLIS_PER_SECOND,
)
from .memory import (
    KvFitInput,
    SeqKvLayout,
    hbm_footprint_per_core_bytes,
    kv_bytes_per_core_parts,
    kv_capacity_and_hit_rate,
)
from .modules import Dtypes, Parallel, StepShape
from .step import FloorOptions, SpecShape, StepFloor, spec_cycle_floor, step_floor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .geometry import ModelGeometry

type _SortBy = Literal["throughput", "interactivity"]


@dataclass(frozen=True, slots=True)
class TrafficScenario:
    """User-selected workload distribution and optional latency SLO bounds."""

    name: str
    isl: int
    osl: int
    prefix_hit_rate: float
    accept_len: float
    think_time_s: float
    concurrencies: tuple[int, ...]
    kv_working_set_turns: float
    prefill_chunk: int = 2048
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    spec_tokens: int | None = None
    kv_offload_gb: float = 0.0

    def __post_init__(self) -> None:
        """Validate input sequence lengths, hit rates, and concurrency parameters."""
        if self.isl <= 0 or self.osl <= 0:
            msg = f"isl and osl must be > 0, got {self.isl}, {self.osl}"
            raise ValueError(msg)
        if (
            not math.isfinite(self.prefix_hit_rate)
            or not 0.0 <= self.prefix_hit_rate <= 1.0
        ):
            msg = f"prefix_hit_rate must be in [0, 1], got {self.prefix_hit_rate}"
            raise ValueError(msg)
        if not math.isfinite(self.accept_len) or self.accept_len < 1.0:
            msg = f"accept_len must be finite and >= 1.0, got {self.accept_len}"
            raise ValueError(msg)
        self._validate_spec_tokens()
        if not math.isfinite(self.think_time_s) or self.think_time_s < 0.0:
            msg = f"think_time_s must be finite and >= 0, got {self.think_time_s}"
            raise ValueError(msg)
        if (
            not math.isfinite(self.kv_working_set_turns)
            or self.kv_working_set_turns < 0.0
        ):
            msg = (
                "kv_working_set_turns must be finite and >= 0, "
                f"got {self.kv_working_set_turns}"
            )
            raise ValueError(msg)
        if not math.isfinite(self.kv_offload_gb) or self.kv_offload_gb < 0.0:
            msg = f"kv_offload_gb must be finite and >= 0, got {self.kv_offload_gb}"
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
        if not self.concurrencies or any(
            concurrency <= 0 for concurrency in self.concurrencies
        ):
            msg = f"concurrencies must be positive ints, got {self.concurrencies}"
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
    moe_dispatch: Literal["allreduce", "ag_ep", "a2a_push", "fused_rs"]
    resident_batch_cap: int | None = None
    residual: Literal["replicated", "sp"] = "replicated"


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
    kv_cache_capacity_tokens: int | None = None
    effective_prefix_hit_rate: float | None = None
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
    sort_by: _SortBy = "throughput"
    model_id: str = ""


@dataclass(frozen=True, slots=True)
class SolveOptions:
    """Search mode, slice budget, HBM reserve, and ordering options."""

    mode: Literal["auto", "mixed", "disagg"] = "auto"
    max_chips: int | None = None
    max_instances: int | None = None
    hbm_reserve_gb: float = DEFAULT_HBM_RESERVE_GB
    topologies: tuple[SliceTopology, ...] | None = None
    collective_latency_s: float = 0.0
    sort_by: _SortBy = "throughput"
    include_pairs: bool = False
    top_k: int | None = None


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


def _candidate_sort_key(point: ParetoPoint, sort_by: _SortBy) -> tuple[object, ...]:
    """Build deterministic per-concurrency sort key for a candidate point."""
    primary, secondary = (
        (point.est_interactivity_tok_s, point.token_throughput_per_chip)
        if sort_by == "interactivity"
        else (point.token_throughput_per_chip, point.est_interactivity_tok_s)
    )
    return (
        not point.slo_met,
        -primary,
        -secondary,
        point.total_chips,
        point.mode,
        _pool_tie_key(point.decode_pool),
        _pool_tie_key(point.prefill_pool),
    )


def _order_and_mark_candidates(
    points: Sequence[ParetoPoint],
    sort_by: _SortBy,
    top_k: int | None,
) -> tuple[tuple[ParetoPoint, ...], tuple[ParetoPoint, ...]]:
    """Mark Pareto members and order all legal candidates per concurrency."""
    if not points:
        return (), ()
    by_conc: dict[int, list[ParetoPoint]] = {}
    for point in points:
        by_conc.setdefault(point.concurrency, []).append(point)
    ordered: list[ParetoPoint] = []
    for concurrency in sorted(by_conc):
        group = by_conc[concurrency]
        flags = _pareto_flags_2d(
            [point_item.token_throughput_per_chip for point_item in group],
            [point_item.est_interactivity_tok_s for point_item in group],
        )
        marked = [
            replace(point_item, is_pareto=flag)
            for point_item, flag in zip(group, flags, strict=True)
        ]
        marked.sort(key=lambda point_item: _candidate_sort_key(point_item, sort_by))
        if top_k is not None:
            marked = marked[:top_k]
        ordered.extend(marked)
    frontier = tuple(point_item for point_item in ordered if point_item.is_pareto)
    return tuple(ordered), frontier


def _pool_record_sort_key(
    record: PoolFloorRecord, sort_by: _SortBy
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
    sort_by: _SortBy,
    top_k: int | None,
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
        if top_k is not None:
            marked = marked[:top_k]
        ordered.extend(marked)
    return tuple(ordered)


@dataclass(frozen=True)
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
    sort_by: _SortBy = "throughput"
    decode_cache: dict[tuple[object, ...], StepFloor] = field(
        default_factory=dict, compare=False, hash=False
    )
    prefill_cache: dict[tuple[object, ...], StepFloor] = field(
        default_factory=dict, compare=False, hash=False
    )
    step_cache: dict[tuple[object, ...], StepFloor] = field(
        default_factory=dict, compare=False, hash=False
    )


@dataclass(frozen=True)
class _PrefillOption:
    """Feasible prefill configuration for disaggregated serving."""

    topology: SliceTopology
    parallel: Parallel
    ttft_ms: float
    floor_ttft_ms: float
    rate_p: float
    hbm_gb: float
    binding: str
    w_base_b: float = 0.0
    kv_bytes_per_tok: float = 0.0
    fixed_kv_per_seq_b: float = 0.0
    offload_b_per_core: float = 0.0


@dataclass(frozen=True)
class _DisaggDecodeState:
    """Decode configuration and memory bounds for disaggregated serving."""

    topology: SliceTopology
    parallel: Parallel
    w_base_b: float
    b_hbm_max: int
    kv_bytes_per_tok: float
    offload_b_per_core: float
    fixed_kv_per_seq_b: float = 0.0
    alt_sp: _DisaggDecodeState | None = None


@dataclass(frozen=True)
class _MixedPointBounds:
    """Precomputed base weights, capacity limits, and TTFT floor for mixed search."""

    w_base_b: float
    kv_bytes_per_tok: float
    b_hbm_max: int
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
    hbm_used_gb: float
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
        hbm_used_gb=spec.hbm_used_gb,
        hbm_limit_gb=spec.hbm_limit_gb,
        moe_dispatch=spec.parallel.moe_dispatch,
        resident_batch_cap=spec.resident_batch_cap,
        residual=spec.parallel.residual,
    )


def _is_strictly_better_point(
    candidate: ParetoPoint,
    incumbent: ParetoPoint,
    sort_by: _SortBy,
) -> bool:
    """Return True iff candidate strictly beats incumbent on SLO + primary metric."""
    if candidate.slo_met != incumbent.slo_met:
        return candidate.slo_met
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


def _is_strictly_better_decode_record(
    candidate: PoolFloorRecord,
    incumbent: PoolFloorRecord,
    sort_by: _SortBy,
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
        return serial_cycle_s, 0.0, stage_floor.binding
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
        ),
    )
    eval_context.prefill_cache[cache_key] = result
    return result


def _clamp_resident_batch(
    eval_context: _SolveContext,
    parallel: Parallel,
    b_start: int,
) -> tuple[int, float] | None:
    """Decrease resident batch until HBM usage fits within hbm_limit_b."""
    resident_batch = b_start
    hbm_b = hbm_footprint_per_core_bytes(
        eval_context.geometry,
        eval_context.dtypes,
        parallel,
        SeqKvLayout(resident_batch, eval_context.ctx_full),
        chip=eval_context.chip,
    )
    while resident_batch > 1 and hbm_b > eval_context.hbm_limit_b:
        resident_batch -= 1
        hbm_b = hbm_footprint_per_core_bytes(
            eval_context.geometry,
            eval_context.dtypes,
            parallel,
            SeqKvLayout(resident_batch, eval_context.ctx_full),
            chip=eval_context.chip,
        )
    if hbm_b > eval_context.hbm_limit_b:
        return None
    return resident_batch, hbm_b


def _find_prefill_options(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[_PrefillOption], float]:
    """Enumerate all legal, HBM-feasible prefill configurations, collapsing residual."""
    prefill_options: list[_PrefillOption] = []
    min_req_ttft_ms = math.inf
    for topology in topos:
        offload_b_per_core = (
            float(eval_context.scenario.kv_offload_gb)
            * UNIT_GIGA
            / float(max(1, topology.num_cores))
        )
        for base_parallel in _candidate_prefill_parallels(
            eval_context.geometry, topology
        ):
            best_opt: _PrefillOption | None = None
            min_mode_floor_ttft_ms = math.inf
            for parallel in _candidate_residual_parallels(
                eval_context.geometry, base_parallel
            ):
                hbm_b = hbm_footprint_per_core_bytes(
                    eval_context.geometry,
                    eval_context.dtypes,
                    parallel,
                    SeqKvLayout(1, eval_context.scenario.isl),
                    chip=eval_context.chip,
                )
                if hbm_b > eval_context.hbm_limit_b:
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
                floor_ttft_ms = (
                    prefill_floor.hard_floor_s * pp_factor * UNIT_MILLIS_PER_SECOND
                )
                min_mode_floor_ttft_ms = min(min_mode_floor_ttft_ms, floor_ttft_ms)
                rate_p = float(max(1, parallel.dp)) / max(
                    1e-9, prefill_floor.serial_estimate_s
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
                cand_opt = _PrefillOption(
                    topology=topology,
                    parallel=parallel,
                    ttft_ms=ttft_ms,
                    floor_ttft_ms=floor_ttft_ms,
                    rate_p=rate_p,
                    hbm_gb=hbm_b / UNIT_GIGA,
                    binding=prefill_floor.binding,
                    w_base_b=w_base_p,
                    kv_bytes_per_tok=kv_b_p,
                    fixed_kv_per_seq_b=fixed_kv_p,
                    offload_b_per_core=offload_b_per_core,
                )
                if best_opt is None or cand_opt.ttft_ms < best_opt.ttft_ms - 1e-12:
                    best_opt = cand_opt
            if best_opt is None:
                continue
            min_req_ttft_ms = min(min_req_ttft_ms, min_mode_floor_ttft_ms)
            prefill_options.append(
                replace(best_opt, floor_ttft_ms=min_mode_floor_ttft_ms)
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
    if eval_context.scenario.think_time_s <= 0.0:
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
            float(concurrency)
            * service_time_s
            / (service_time_s + eval_context.scenario.think_time_s)
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
    resident_batch, hbm_b = clamped
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
    server_cap_rate = res_cap / max(1e-9, total_req_work_s)
    mix_step_wait_s = mix_verify_serial_s if concurrency > 1 else 0.0
    ttft_in_server_s = max(
        sf_ttft_c.serial_estimate_s * pp_factor + mix_step_wait_s,
        prefill_backlog_s,
    )
    service_time_final_s = ttft_in_server_s + dec_steps_per_req * effective_step_s
    rate_req_s = (
        min(
            server_cap_rate,
            float(concurrency)
            / max(1e-9, service_time_final_s + eval_context.scenario.think_time_s),
        )
        if eval_context.scenario.think_time_s > 0.0
        else c_served / max(1e-9, total_req_work_s)
    )
    est_tpot_ms = (
        effective_step_s * UNIT_MILLIS_PER_SECOND / eval_context.effective_accept_len
    )
    c_demand = min(float(concurrency), float(local_batch * max(1, parallel.dp)))
    queue_wait_s = (
        max(0.0, (c_demand / max(1e-9, res_cap)) - 1.0) * service_time_final_s
    )
    est_ttft_ms = (ttft_in_server_s + queue_wait_s) * UNIT_MILLIS_PER_SECOND
    interactivity = UNIT_MILLIS_PER_SECOND / max(1e-6, est_tpot_ms)
    out_tput = rate_req_s * float(eval_context.scenario.osl) / float(topology.num_chips)
    tot_tput = (
        rate_req_s
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
        / float(topology.num_chips)
    )
    slo_ok = (
        eval_context.scenario.ttft_slo_ms is None
        or est_ttft_ms <= eval_context.scenario.ttft_slo_ms
    ) and (
        eval_context.scenario.tpot_slo_ms is None
        or est_tpot_ms <= eval_context.scenario.tpot_slo_ms
    )
    pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="mixed",
            instances=1,
            topology=topology,
            parallel=parallel,
            hbm_used_gb=hbm_b / UNIT_GIGA,
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
        slo_met=slo_ok,
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=eff_hit_rate,
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
        for base_parallel in _candidate_decode_parallels(
            eval_context.geometry, topology
        ):
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
                pp_factor = 1.0 + float(parallel.pp - 1) / float(
                    max(1, eval_context.n_chunks)
                )
                min_req_ttft_ms = min(
                    min_req_ttft_ms,
                    prefill_step_floor.hard_floor_s
                    * pp_factor
                    * UNIT_MILLIS_PER_SECOND,
                )
                fitting_modes.append(
                    (
                        parallel,
                        _MixedPointBounds(
                            w_base_b=w_base_b,
                            kv_bytes_per_tok=kv_bytes_core,
                            b_hbm_max=b_hbm_max,
                            sf_ttft=prefill_step_floor,
                            fixed_kv_per_seq_b=fixed_kv_per_seq_b,
                        ),
                    )
                )
            if not fitting_modes:
                continue
            for concurrency in eval_context.scenario.concurrencies:
                best_point: ParetoPoint | None = None
                for parallel, bounds in fitting_modes:
                    point = _evaluate_mixed_point(
                        eval_context, topology, parallel, concurrency, bounds
                    )
                    if point is None:
                        continue
                    if best_point is None or _is_strictly_better_point(
                        point, best_point, eval_context.sort_by
                    ):
                        best_point = point
                if best_point is not None:
                    candidates.append(best_point)
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
                hbm_used_gb=opt.hbm_gb,
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
    resident_batch, hbm_d = clamped
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
            hbm_used_gb=hbm_d / UNIT_GIGA,
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
        rate_d_cap = c_served_d / (dec_steps_per_req * max(1e-9, decode_serial_s))
        c_demand_d = min(
            float(target.concurrency),
            float(
                batch_stats.local_batch
                * max(1, target.decode_instances * dec.parallel.dp)
            ),
        )
        queue_wait_d_ms = (
            max(0.0, (c_demand_d / max(1e-9, res_cap_d)) - 1.0)
            * (dec_steps_per_req * decode_serial_s)
            * UNIT_MILLIS_PER_SECOND
        )
    prefill_rate = float(target.prefill_instances) * batch_stats.rate_p_eff
    if eval_context.scenario.think_time_s > 0.0:
        client_rate = float(target.concurrency) / max(
            1e-9,
            batch_stats.ttft_eff_ms / UNIT_MILLIS_PER_SECOND
            + dec_steps_per_req * decode_serial_s
            + eval_context.scenario.think_time_s,
        )
        coupled_rate = min(prefill_rate, rate_d_cap, client_rate)
    else:
        coupled_rate = min(prefill_rate, rate_d_cap)
    ll_ttft_s = max(
        0.0,
        float(target.concurrency) / max(1e-9, coupled_rate)
        - dec_steps_per_req * decode_serial_s
        - eval_context.scenario.think_time_s,
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
    if eval_context.scenario.think_time_s <= 0.0:
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
            / (service_time_s + eval_context.scenario.think_time_s)
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
    clamped = _clamp_resident_batch(eval_context, dec.parallel, b_start)
    if clamped is None:
        return None
    resident_batch, hbm_d = clamped
    working_set_decode = max(
        1,
        math.ceil(
            target.concurrency / max(1, target.decode_instances * dec.parallel.dp)
        ),
    )
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
    working_set_prefill = max(
        1,
        math.ceil(
            target.concurrency / max(1, target.prefill_instances * pref.parallel.dp)
        ),
    )
    _, eff_hit_prefill = kv_capacity_and_hit_rate(
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
    slo_ok = (
        eval_context.scenario.ttft_slo_ms is None
        or est_ttft_d_ms <= eval_context.scenario.ttft_slo_ms
    ) and (
        eval_context.scenario.tpot_slo_ms is None
        or est_tpot_ms <= eval_context.scenario.tpot_slo_ms
    )
    p_pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="prefill",
            instances=target.prefill_instances,
            topology=pref.topology,
            parallel=pref.parallel,
            hbm_used_gb=pref.hbm_gb,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            resident_batch_cap=None,
        )
    )
    d_pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="decode",
            instances=target.decode_instances,
            topology=dec.topology,
            parallel=dec.parallel,
            hbm_used_gb=hbm_d / UNIT_GIGA,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            resident_batch_cap=resident_batch,
        )
    )
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
        slo_met=slo_ok,
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=eff_hit_prefill,
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
        return rep_point
    sp_point = _evaluate_single_disagg_pair(
        eval_context, dec.alt_sp, pref, target, cache
    )
    if rep_point is None:
        return sp_point
    if sp_point is None:
        return rep_point
    if _is_strictly_better_point(sp_point, rep_point, eval_context.sort_by):
        return sp_point
    return rep_point


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
        float(concurrency) / (dec_steps_per_req * max(1e-9, dec_one_serial_s))
        if dec_steps_per_req > 0.0
        else math.inf
    )
    eff_max_inst = (
        eval_context.max_instances
        if eval_context.max_instances is not None
        else DEFAULT_MAX_INSTANCES
    )
    if eval_context.max_chips is None:
        if math.isfinite(rate_d_one):
            ratio_xp = math.ceil(rate_d_one / max(1e-9, pref.rate_p)) + 1
            ratio_yd = math.ceil(pref.rate_p / max(1e-9, rate_d_one)) + 1
        else:
            ratio_xp = eff_max_inst
            ratio_yd = 1
        max_xp = min(eff_max_inst, max(1, ratio_xp))
        max_yd = min(eff_max_inst, max(1, ratio_yd))
    else:
        max_xp = min(
            eff_max_inst, max(1, eval_context.max_chips // pref.topology.num_chips)
        )
        max_yd = min(
            eff_max_inst, max(1, eval_context.max_chips // dec.topology.num_chips)
        )
    points: list[ParetoPoint] = []
    for prefill_instances in range(1, max_xp + 1):
        for decode_instances in range(1, max_yd + 1):
            total_chips = (
                prefill_instances * pref.topology.num_chips
                + decode_instances * dec.topology.num_chips
            )
            if (
                eval_context.max_chips is not None
                and total_chips > eval_context.max_chips
            ):
                continue
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


def _validate_solve_options(opts: SolveOptions) -> None:
    """Validate SolveOptions numeric and enum invariants."""
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


def solve_serving_plan(
    geometry: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    options: SolveOptions | None = None,
) -> ServingPlan:
    """Evaluate all physics-legal layouts, mark Pareto members, and order per pool."""
    opts = options or SolveOptions()
    _validate_solve_options(opts)
    topos = _resolve_solve_topologies(opts, chip=chip)
    hbm_limit_b = max(0.0, float(chip.hbm_bytes) - opts.hbm_reserve_gb * UNIT_GIGA)
    effective_accept_len = max(1.0, scenario.accept_len)
    spec_k = (
        scenario.spec_tokens
        if scenario.spec_tokens is not None
        else (
            max(geometry.mtp_layers, math.ceil(effective_accept_len) - 1)
            if effective_accept_len > 1.0
            else 0
        )
    )
    ctx_full = scenario.isl + scenario.osl
    eff_isl = max(1, math.ceil((1.0 - scenario.prefix_hit_rate) * scenario.isl))
    n_chunks = max(1, math.ceil(eff_isl / max(1, scenario.prefill_chunk)))

    eval_context = _SolveContext(
        geometry=geometry,
        dtypes=dtypes,
        chip=chip,
        scenario=scenario,
        mode=opts.mode,
        max_chips=opts.max_chips,
        max_instances=opts.max_instances,
        hbm_reserve_gb=opts.hbm_reserve_gb,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / UNIT_GIGA,
        effective_accept_len=effective_accept_len,
        spec_k=spec_k,
        ctx_full=ctx_full,
        ctx_mid=scenario.isl + scenario.osl // 2,
        eff_isl=eff_isl,
        n_chunks=n_chunks,
        c_max=max(scenario.concurrencies),
        shared_prefix_tokens=round(scenario.prefix_hit_rate * scenario.isl),
        collective_latency_s=opts.collective_latency_s,
        sort_by=opts.sort_by,
    )

    prefill_options, min_prefill_ttft_ms = (
        _find_prefill_options(eval_context, topos)
        if opts.mode in ("auto", "disagg")
        else ([], math.inf)
    )
    mixed_candidates, min_c1_tpot_ms, min_mixed_ttft_ms = _evaluate_mixed_candidates(
        eval_context, topos
    )
    decode_states = (
        _build_decode_pool_states(eval_context, topos)
        if opts.mode in ("auto", "disagg")
        else []
    )
    prefill_pools = _order_and_mark_pool_records(
        _build_prefill_pool_records(eval_context, prefill_options),
        opts.sort_by,
        opts.top_k,
    )
    decode_pools = _order_and_mark_pool_records(
        _build_decode_pool_records(eval_context, decode_states),
        opts.sort_by,
        opts.top_k,
    )
    disagg_candidates = (
        _evaluate_disaggregated_candidates(eval_context, decode_states, prefill_options)
        if opts.include_pairs
        else []
    )
    ordered_candidates, frontier = _order_and_mark_candidates(
        mixed_candidates + disagg_candidates,
        opts.sort_by,
        opts.top_k,
    )
    pools_meet_slo = (
        not prefill_pools or any(record.slo_met for record in prefill_pools)
    ) and (not decode_pools or any(record.slo_met for record in decode_pools))
    warnings = _collect_warnings(
        geometry,
        scenario,
        (
            min_c1_tpot_ms,
            _active_mode_min_ttft(opts.mode, min_prefill_ttft_ms, min_mixed_ttft_ms),
        ),
        ordered_candidates,
        has_any_pool=bool(prefill_pools or decode_pools),
        pools_meet_slo=pools_meet_slo,
        disagg_pairs_skipped=opts.mode in ("auto", "disagg") and not opts.include_pairs,
    )
    slo_met_candidates = [point for point in ordered_candidates if point.slo_met]
    recommended = (
        min(
            slo_met_candidates,
            key=lambda point: _candidate_sort_key(point, opts.sort_by),
        )
        if slo_met_candidates
        else None
    )
    return ServingPlan(
        model_layers=geometry.num_layers,
        chip=chip.name,
        scenario=scenario,
        mode=opts.mode,
        recommended=recommended,
        frontier=frontier,
        warnings=warnings,
        candidates=ordered_candidates,
        prefill_pools=prefill_pools,
        decode_pools=decode_pools,
        hbm_reserve_gb=opts.hbm_reserve_gb,
        collective_latency_s=opts.collective_latency_s,
        spec_k=spec_k,
        sort_by=opts.sort_by,
        model_id=geometry.model_id,
    )
