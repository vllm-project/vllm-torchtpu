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
"""Unconstrained single-pool mixed and xP/yD disaggregated TPU mesh solver."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from .chips import TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import submesh_for_group
from .constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_INSTANCES,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    MIN_MOE_INTERMEDIATE_ALIGNMENT,
    MIN_SLICE_CHIPS,
    UNIT_GIGA,
    UNIT_MICROS_PER_SECOND,
    UNIT_MILLIS_PER_SECOND,
)
from .memory import (
    HbmBreakdown,
    KvFitInput,
    SeqKvLayout,
    default_rpa_page_size,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    kv_capacity_and_hit_rate,
    smem_max_seqs,
)
from .modules import Dtypes, Parallel, StepShape
from .partition import (
    count_full_attn_layers,
)
from .step import FloorOptions, SpecShape, StepFloor, spec_cycle_floor, step_floor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .geometry import ModelGeometry


@dataclass(frozen=True, slots=True)
class TrafficScenario:
    """User-selected workload distribution and optional p90 SLO bounds."""

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
    page_size: int | None = None
    max_num_seqs: int | None = None

    def __post_init__(self) -> None:
        """Validate input sequence lengths, hit rates, and concurrency parameters."""
        if self.isl <= 0 or self.osl <= 0:
            msg = f"isl and osl must be > 0, got {self.isl}, {self.osl}"
            raise ValueError(msg)
        if not 0.0 <= self.prefix_hit_rate <= 1.0:
            msg = f"prefix_hit_rate must be in [0, 1], got {self.prefix_hit_rate}"
            raise ValueError(msg)
        if self.accept_len <= 0.0:
            msg = f"accept_len must be > 0, got {self.accept_len}"
            raise ValueError(msg)
        if self.think_time_s < 0.0:
            msg = f"think_time_s must be >= 0, got {self.think_time_s}"
            raise ValueError(msg)
        if not self.concurrencies or any(c <= 0 for c in self.concurrencies):
            msg = f"concurrencies must be positive ints, got {self.concurrencies}"
            raise ValueError(msg)
        if self.page_size is not None and self.page_size <= 0:
            msg = f"page_size must be > 0, got {self.page_size}"
            raise ValueError(msg)
        if self.max_num_seqs is not None and self.max_num_seqs <= 0:
            msg = f"max_num_seqs must be > 0, got {self.max_num_seqs}"
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
    ovr = overrides or ScenarioOverrides()
    tier_ttft, tier_tpot = EXAMPLE_SLO_TIERS[slo_tier]
    return replace(
        base,
        isl=base.isl if ovr.isl is None else ovr.isl,
        osl=base.osl if ovr.osl is None else ovr.osl,
        concurrencies=(
            base.concurrencies if ovr.concurrencies is None else ovr.concurrencies
        ),
        prefix_hit_rate=(
            base.prefix_hit_rate if ovr.prefix_hit_rate is None else ovr.prefix_hit_rate
        ),
        accept_len=base.accept_len if ovr.accept_len is None else ovr.accept_len,
        think_time_s=(
            base.think_time_s if ovr.think_time_s is None else ovr.think_time_s
        ),
        prefill_chunk=(
            base.prefill_chunk if ovr.prefill_chunk is None else ovr.prefill_chunk
        ),
        ttft_slo_ms=(
            ovr.ttft_slo_ms
            if ovr.ttft_slo_ms is not None
            else (tier_ttft if tier_ttft is not None else base.ttft_slo_ms)
        ),
        tpot_slo_ms=(
            ovr.tpot_slo_ms
            if ovr.tpot_slo_ms is not None
            else (tier_tpot if tier_tpot is not None else base.tpot_slo_ms)
        ),
        kv_working_set_turns=(
            base.kv_working_set_turns
            if ovr.kv_working_set_turns is None
            else ovr.kv_working_set_turns
        ),
    )


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


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Search funnel: (layout, concurrency) tuples inside the chip budget."""

    enumerated: int
    hbm_fit: int
    slo_met: int
    frontier: int


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
    mesh_axes: tuple[int, ...]
    hbm_used_gb: float
    hbm_limit_gb: float
    moe_dispatch: Literal["allreduce", "a2a_push", "ag_ep", "fused_rs"] = "allreduce"
    hbm_weights_gb: float = 0.0
    hbm_kv_gb: float = 0.0
    hbm_state_gb: float = 0.0
    max_num_seqs: int | None = None
    max_model_len: int | None = None


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """Evaluated serving configuration with latency, throughput, and SLO state."""

    mode: Literal["mixed", "disagg"]
    concurrency: int
    total_chips: int
    prefill_pool: PoolMeshPlan | None
    decode_pool: PoolMeshPlan
    p90_ttft_ms: float
    p90_tpot_ms: float
    p90_interactivity_tok_s: float
    output_throughput_per_chip: float
    token_throughput_per_chip: float
    p_mixed: float
    binding: str
    slo_met: bool
    decode_floor: FloorTimes | None = None
    kv_cache_capacity_tokens: int | None = None
    effective_prefix_hit_rate: float | None = None

    @property
    def est_ttft_ms(self) -> float:
        """Compatibility property for p90_ttft_ms."""
        return self.p90_ttft_ms

    @property
    def est_tpot_ms(self) -> float:
        """Compatibility property for p90_tpot_ms."""
        return self.p90_tpot_ms

    @property
    def est_interactivity_tok_s(self) -> float:
        """Compatibility property for p90_interactivity_tok_s."""
        return self.p90_interactivity_tok_s


@dataclass(frozen=True, slots=True)
class ServingPlan:
    """Full Pareto frontier and recommended serving configuration for a scenario."""

    model_layers: int
    chip: str
    scenario: TrafficScenario
    mode: str
    recommended: ParetoPoint | None
    frontier: tuple[ParetoPoint, ...]
    warnings: tuple[str, ...]
    candidates: tuple[ParetoPoint, ...] = ()
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    search: SearchStats | None = None


@dataclass(frozen=True, slots=True)
class SolveOptions:
    """Search mode, slice budget, and HBM headroom options for solve_serving_plan."""

    mode: Literal["auto", "mixed", "disagg"] = "auto"
    max_chips: int | None = None
    max_instances: int | None = None
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    topologies: tuple[SliceTopology, ...] | None = None


def _divisors(n: int) -> tuple[int, ...]:
    """Return all integer divisors of n in ascending order."""
    return tuple(d for d in range(1, n + 1) if n % d == 0)


def _valid_head_tp(geom: ModelGeometry, tp: int) -> bool:
    """Check whether TP divides attention heads or evenly groups KV heads."""
    if geom.num_heads > 0 and (tp > geom.num_heads or geom.num_heads % tp != 0):
        return False
    if geom.num_kv_heads <= 0:
        return True
    return geom.num_kv_heads % tp == 0 or tp % geom.num_kv_heads == 0


def _has_gdn_layers(geom: ModelGeometry) -> bool:
    """Check whether any layer type in geometry belongs to gated delta net."""
    return any(
        tok in ("linear_attention", "gdn", "gated_deltanet", "mamba")
        for tok in geom.layer_types
    )


def _valid_gdn_degree(geom: ModelGeometry, deg: int) -> bool:
    """Check legality of GDN degree across key and value heads."""
    if deg <= 1 or not _has_gdn_layers(geom):
        return True
    if geom.linear_v_heads <= 0 or geom.linear_k_heads <= 0:
        return True
    if geom.linear_v_heads % deg != 0:
        return False
    uq = min(geom.linear_k_heads, deg)
    if geom.linear_k_heads % uq != 0 or deg % uq != 0:
        return False
    return (geom.linear_v_heads // deg) % (geom.linear_k_heads // uq) == 0


def _valid_raiden_gdn_pair(
    geom: ModelGeometry,
    par_p: Parallel,
    par_d: Parallel,
) -> bool:
    """Check whether prefill and decode parallelisms form a legal GDN pair."""
    if not _has_gdn_layers(geom):
        return True
    if geom.linear_v_heads <= 0 or geom.linear_k_heads <= 0:
        return True
    deg_p = par_p.tp * par_p.pcp
    deg_d = par_d.tp * par_d.pcp
    if not (_valid_gdn_degree(geom, deg_p) and _valid_gdn_degree(geom, deg_d)):
        return False
    if par_p.pp > 1 and deg_p > 1 and deg_d > deg_p and deg_d % deg_p == 0:
        return True
    kq = geom.linear_k_heads
    if max(1, deg_p // kq) != max(1, deg_d // kq):
        return False
    return deg_p == 1 or deg_p % deg_d == 0


def _valid_raiden_kv_pair(
    geom: ModelGeometry,
    prefill_parallel: Parallel,
    decode_parallel: Parallel,
) -> bool:
    """Check whether prefill and decode parallelisms form a legal GQA/MLA KV pair."""
    if geom.num_kv_heads <= 0:
        return True
    deg_p = max(1, prefill_parallel.tp * prefill_parallel.pcp)
    deg_d = max(1, decode_parallel.tp * decode_parallel.pcp)
    return deg_p % deg_d == 0 or deg_d % deg_p == 0


def _candidate_moe_dispatches(
    moe_mode: Literal["tp_slice", "ep"],
    dp: int,
) -> list[Literal["allreduce", "ag_ep", "a2a_push"]]:
    """Return legal MoE decode dispatch collectives for mode and DP degree."""
    if moe_mode == "ep" and dp > 1:
        return ["ag_ep", "a2a_push"]
    return ["allreduce"]


def _candidate_decode_parallels(
    geom: ModelGeometry,
    topo: SliceTopology,
) -> list[Parallel]:
    """Enumerate legal decode parallelism configurations for a given topology."""
    cores = topo.num_cores
    out: list[Parallel] = []
    has_kv_cache = geom.num_kv_heads > 0 or geom.kv_lora_rank > 0
    for tp in _divisors(cores):
        if not _valid_head_tp(geom, tp) or not _valid_gdn_degree(geom, tp):
            continue
        dp = cores // tp
        if not has_kv_cache:
            kv_splits: tuple[int, ...] = (1,)
        else:
            kv_group = (
                max(1, tp // geom.num_kv_heads)
                if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
                else tp
            )
            kv_splits = _divisors(kv_group) if kv_group > 1 else (1,)
        moe_mode: Literal["tp_slice", "ep"] = (
            "ep"
            if geom.num_experts > 0 and cores > 1 and geom.num_experts >= cores
            else "tp_slice"
        )
        if geom.num_experts > 0 and moe_mode == "tp_slice" and dp > 1:
            continue
        ep_val = cores if moe_mode == "ep" else 1
        mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
        modes: list[
            tuple[
                Literal["tp_slice", "ep"],
                int,
                Literal["allreduce", "ag_ep", "a2a_push"],
            ]
        ] = [
            (moe_mode, ep_val, disp) for disp in _candidate_moe_dispatches(moe_mode, dp)
        ]
        if (
            moe_mode == "ep"
            and dp == 1
            and geom.moe_inter // tp >= MIN_MOE_INTERMEDIATE_ALIGNMENT
            and (geom.moe_inter // tp) % MIN_MOE_INTERMEDIATE_ALIGNMENT == 0
        ):
            modes.append(("tp_slice", 1, "allreduce"))
        for kvs in kv_splits:
            for m_mode, m_ep, m_disp in modes:
                out.append(
                    Parallel(
                        tp=tp,
                        kv_split=kvs,
                        moe=m_mode,
                        moe_dispatch=m_disp,
                        ranks_per_chip=topo.cores_per_chip,
                        dp=dp,
                        ep=m_ep,
                        pp=1,
                        pcp=1,
                        mesh_axes=mesh_ax,
                    ),
                )
    return out


def _candidate_prefill_parallels(
    geom: ModelGeometry,
    topo: SliceTopology,
    *,
    for_disagg: bool = False,
) -> list[Parallel]:
    """Enumerate legal prefill parallelism configurations for a given topology."""
    cores = topo.num_cores
    n_full_attn = count_full_attn_layers(geom)
    out: list[Parallel] = []
    for pp in _divisors(cores):
        if geom.num_layers < pp:
            continue
        if for_disagg and 0 < n_full_attn < geom.num_layers and pp > n_full_attn:
            continue
        rem = cores // pp
        for pcp in _divisors(rem):
            tp = rem // pcp
            # Runtime supports PP>1 with PCP>1 (standalone PP+TP is blocked).
            if tp < 1 or (pp > 1 and tp > 1):
                continue
            if not _valid_head_tp(geom, tp) or not _valid_gdn_degree(geom, tp * pcp):
                continue
            moe_mode, ep_val = _resolve_prefill_moe(geom, tp, pcp)
            if moe_mode is None:
                continue
            mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
            dispatches: tuple[Literal["allreduce", "ag_ep"], ...] = (
                ("ag_ep",)
                if (moe_mode == "ep" and tp == 1)
                else (("allreduce", "ag_ep") if moe_mode == "ep" else ("allreduce",))
            )
            out.extend(
                Parallel(
                    tp=tp,
                    kv_split=1,
                    moe=moe_mode,
                    moe_dispatch=disp,
                    ranks_per_chip=topo.cores_per_chip,
                    dp=1,
                    ep=ep_val,
                    pp=pp,
                    pcp=pcp,
                    mesh_axes=mesh_ax,
                )
                for disp in dispatches
            )
    return out


def _resolve_prefill_moe(
    geom: ModelGeometry,
    tp: int,
    pcp: int,
) -> tuple[Literal["tp_slice", "ep"] | None, int]:
    """Determine (moe_mode, ep_degree) for prefill candidates, or None if invalid."""
    if pcp > 1 and geom.num_experts > 0:
        ep_span = tp * pcp
        if geom.num_experts < ep_span or geom.num_experts % ep_span != 0:
            return None, 1
        return "ep", ep_span
    moe_mode: Literal["tp_slice", "ep"] = (
        "ep"
        if geom.num_experts > 0 and tp > 1 and geom.num_experts >= tp
        else "tp_slice"
    )
    return moe_mode, (tp if moe_mode == "ep" else 1)


def _pareto_filter(points: Sequence[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Filter points to non-dominated Pareto frontier sorted by concurrency."""
    if not points:
        return ()
    frontier: list[ParetoPoint] = []
    for p in points:
        dominated = any(
            q.concurrency <= p.concurrency
            and q.output_throughput_per_chip >= p.output_throughput_per_chip
            and q.p90_interactivity_tok_s >= p.p90_interactivity_tok_s
            and (
                q.concurrency < p.concurrency
                or q.output_throughput_per_chip > p.output_throughput_per_chip
                or q.p90_interactivity_tok_s > p.p90_interactivity_tok_s
            )
            for q in points
        )
        if not dominated:
            frontier.append(p)
    frontier.sort(
        key=lambda x: (
            x.concurrency,
            -x.output_throughput_per_chip,
            -x.p90_interactivity_tok_s,
        ),
    )
    return tuple(frontier)


@dataclass
class _SolveContext:
    """Internal solver context capturing model, hardware, and scenario constants."""

    geom: ModelGeometry
    dtypes: Dtypes
    chip: ChipSpec
    scenario: TrafficScenario
    mode: str
    max_chips: int | None
    max_instances: int | None
    gpu_memory_utilization: float
    hbm_limit_b: float
    hbm_limit_gb: float
    eff_al: float
    spec_k: int
    ctx_full: int
    ctx_mid: int
    eff_isl: int
    n_chunks: int
    page_size: int
    c_max: int
    enumerated: int = 0
    hbm_fit: int = 0


@dataclass(frozen=True)
class _PrefillOption:
    """Feasible prefill configuration for disaggregated serving."""

    topo: SliceTopology
    par: Parallel
    ttft_ms: float
    rate_p: float
    hbm_gb: float
    hbm_bd: HbmBreakdown


@dataclass(frozen=True)
class _DisaggDecodeState:
    """Decode configuration and memory bounds for disaggregated serving."""

    topo: SliceTopology
    par: Parallel
    w_base_b: float
    b_hbm_max: int
    b_smem_max: int
    b_kv_phys: float
    offload_b_per_core: float


@dataclass(frozen=True)
class _MixedPointBounds:
    """Precomputed base weights, capacity limits, and TTFT floor for mixed search."""

    w_base_b: float
    b_kv_phys_core: float
    b_hbm_max: int
    b_smem_max: int
    sf_ttft: StepFloor


@dataclass(frozen=True)
class _DisaggRatioTarget:
    """Concurrency and instance counts (xP, yD) for a disaggregated point."""

    c: int
    x_p: int
    y_d: int
    total_chips: int


@dataclass(frozen=True)
class _DisaggBatchStats:
    """Resolved decode batch size and effective prefill rate for a ratio target."""

    b_res: int
    b_local: int
    ttft_eff_ms: float
    rate_p_eff: float


@dataclass(frozen=True)
class _PoolPlanInput:
    """Inputs needed to materialize one PoolMeshPlan instance pool."""

    role: Literal["mixed", "prefill", "decode"]
    instances: int
    topo: SliceTopology
    par: Parallel
    hbm_used_gb: float
    hbm_limit_gb: float
    max_num_seqs: int | None
    max_model_len: int
    hbm_weights_gb: float = 0.0
    hbm_kv_gb: float = 0.0
    hbm_state_gb: float = 0.0


def _floor_times(sf: StepFloor) -> FloorTimes:
    """Extract decode-step resource times (ms) from a StepFloor."""
    return FloorTimes(
        hbm_ms=sf.hbm_s * UNIT_MILLIS_PER_SECOND,
        matmul_ms=sf.matmul_s * UNIT_MILLIS_PER_SECOND,
        ici_ms=sf.ici_s * UNIT_MILLIS_PER_SECOND,
        chain_ms=sf.chain_s * UNIT_MILLIS_PER_SECOND,
        hop_latency_ms=sf.lat_s * UNIT_MILLIS_PER_SECOND,
        hard_floor_ms=sf.hard_floor_s * UNIT_MILLIS_PER_SECOND,
        binding=sf.binding,
    )


def _build_pool_mesh_plan(spec: _PoolPlanInput) -> PoolMeshPlan:
    """Construct a PoolMeshPlan from topology, Parallel, and HBM usage."""
    axes = (
        spec.par.mesh_axes
        if spec.par.mesh_axes is not None
        else submesh_for_group(spec.par.tp, spec.topo, spec.topo.cores_per_chip)
    )
    return PoolMeshPlan(
        role=spec.role,
        instances=spec.instances,
        topology=spec.topo.name,
        chips_per_instance=spec.topo.num_chips,
        tp=spec.par.tp,
        dp=spec.par.dp,
        ep=spec.par.ep,
        pp=spec.par.pp,
        pcp=spec.par.pcp,
        kv_split=spec.par.kv_split,
        mesh_axes=axes,
        hbm_used_gb=spec.hbm_used_gb,
        hbm_limit_gb=spec.hbm_limit_gb,
        moe_dispatch=spec.par.moe_dispatch,
        hbm_weights_gb=spec.hbm_weights_gb,
        hbm_kv_gb=spec.hbm_kv_gb,
        hbm_state_gb=spec.hbm_state_gb,
        max_num_seqs=spec.max_num_seqs,
        max_model_len=spec.max_model_len,
    )


def _eval_decode_floor(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    batch: int,
) -> StepFloor:
    """Evaluate single-step or speculative-cycle floor at midpoint context."""
    opts = FloorOptions(dt=eval_context.dtypes, topology=topo)
    if eval_context.spec_k > 0:
        return spec_cycle_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            SpecShape(batch=batch, context=eval_context.ctx_mid, k=eval_context.spec_k),
            opts,
        )
    return step_floor(
        eval_context.geom,
        eval_context.chip,
        par,
        StepShape(batch=batch, context=eval_context.ctx_mid),
        opts,
    )


def _prefill_step_floor(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    prefix_hit_rate: float,
) -> StepFloor:
    """Evaluate prompt prefill floor for a given topology, parallel, and hit rate."""
    return step_floor(
        eval_context.geom,
        eval_context.chip,
        par,
        StepShape(
            batch=0,
            rows_per_seq=0,
            context=0,
            state_writes=0,
            prefill_tokens=eval_context.scenario.isl,
            prefix_hit_rate=prefix_hit_rate,
        ),
        FloorOptions(dt=eval_context.dtypes, topology=topo),
    )


def _clamp_resident_batch(
    eval_context: _SolveContext,
    par: Parallel,
    b_start: int,
) -> tuple[int, HbmBreakdown] | None:
    """Decrease resident batch until HBM usage fits within hbm_limit_b."""
    b_res = b_start
    hbm_bd = hbm_breakdown_per_core_bytes(
        eval_context.geom,
        eval_context.dtypes,
        par,
        SeqKvLayout(b_res, eval_context.ctx_full),
    )
    while b_res > 1 and hbm_bd.total > eval_context.hbm_limit_b:
        b_res -= 1
        hbm_bd = hbm_breakdown_per_core_bytes(
            eval_context.geom,
            eval_context.dtypes,
            par,
            SeqKvLayout(b_res, eval_context.ctx_full),
        )
    if hbm_bd.total > eval_context.hbm_limit_b:
        return None
    return b_res, hbm_bd


def _find_prefill_options(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[_PrefillOption], float]:
    """Find fastest legal prefill configurations grouped by GDN compatibility class."""
    prefill_options: list[_PrefillOption] = []
    min_req_ttft_ms = math.inf
    for topo in topos:
        best_by_class: dict[tuple[int, bool], _PrefillOption] = {}
        for par in _candidate_prefill_parallels(
            eval_context.geom, topo, for_disagg=True
        ):
            hbm_bd = hbm_breakdown_per_core_bytes(
                eval_context.geom,
                eval_context.dtypes,
                par,
                SeqKvLayout(1, eval_context.scenario.isl),
            )
            hbm_b = hbm_bd.total
            if hbm_b > eval_context.hbm_limit_b:
                continue
            sf_p = _prefill_step_floor(
                eval_context, topo, par, eval_context.scenario.prefix_hit_rate
            )
            ttft_ms = (
                sf_p.serial_estimate_s
                * (1.0 + float(par.pp - 1) / float(max(1, eval_context.n_chunks)))
                * UNIT_MILLIS_PER_SECOND
            )
            min_req_ttft_ms = min(min_req_ttft_ms, ttft_ms)
            rate_p = 1.0 / max(1e-9, sf_p.serial_estimate_s)
            cls_key = (par.tp * par.pcp, par.pp > 1)
            prev = best_by_class.get(cls_key)
            if prev is None or rate_p > prev.rate_p:
                best_by_class[cls_key] = _PrefillOption(
                    topo=topo,
                    par=par,
                    ttft_ms=ttft_ms,
                    rate_p=rate_p,
                    hbm_gb=hbm_b / UNIT_GIGA,
                    hbm_bd=hbm_bd,
                )
        prefill_options.extend(best_by_class.values())
    return prefill_options, min_req_ttft_ms


def _solve_mixed_think_batch(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    c: int,
    sf_ttft: StepFloor,
) -> int:
    """Resolve Little's-law active local batch size under non-zero think time."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    b_local = max(1, math.ceil(c / max(1, par.dp)))
    if eval_context.scenario.think_time_s <= 0.0:
        return b_local
    for _ in range(2):
        sf_dec_tmp = _eval_decode_floor(eval_context, topo, par, b_local)
        w_srv = (
            sf_ttft.serial_estimate_s + dec_steps_per_req * sf_dec_tmp.serial_estimate_s
        )
        c_act = float(c) * w_srv / (w_srv + eval_context.scenario.think_time_s)
        b_local = max(1, math.ceil(c_act / max(1, par.dp)))
    return b_local


def _fit_mixed_resident_batch(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    c: int,
    bounds: _MixedPointBounds,
) -> tuple[int, int, HbmBreakdown] | None:
    """Compute active local batch and HBM-fitting resident batch."""
    b_local = _solve_mixed_think_batch(eval_context, topo, par, c, bounds.sf_ttft)
    b_start = min(b_local, bounds.b_hbm_max, bounds.b_smem_max)
    if eval_context.scenario.max_num_seqs is not None:
        b_start = min(b_start, eval_context.scenario.max_num_seqs)
    clamped = _clamp_resident_batch(eval_context, par, b_start)
    if clamped is None:
        return None
    b_res, hbm_bd = clamped
    return b_local, b_res, hbm_bd


def _evaluate_mixed_point(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    c: int,
    bounds: _MixedPointBounds,
) -> ParetoPoint | None:
    """Evaluate one concurrency target for a single-pool mixed serving configuration."""
    if bounds.b_hbm_max < 1 or bounds.b_smem_max < 1:
        return None
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    offload_b_per_core = (
        float(eval_context.scenario.kv_offload_gb)
        * UNIT_GIGA
        / float(max(1, topo.num_cores))
    )
    fitted = _fit_mixed_resident_batch(eval_context, topo, par, c, bounds)
    if fitted is None:
        return None
    b_local, b_res, hbm_bd = fitted
    hbm_b = hbm_bd.total
    eval_context.hbm_fit += 1
    cap_tok, h_eff = kv_capacity_and_hit_rate(
        eval_context.geom,
        eval_context.dtypes,
        par,
        b_res,
        KvFitInput(
            b_ws=b_local,
            ctx_full=eval_context.ctx_full,
            hbm_limit_b=eval_context.hbm_limit_b,
            w_base_b=bounds.w_base_b,
            b_kv_phys_core=bounds.b_kv_phys_core,
            offload_b_per_core=offload_b_per_core,
            prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
            kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
        ),
    )
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    sf_ttft_c = (
        _prefill_step_floor(eval_context, topo, par, h_eff)
        if h_eff != eval_context.scenario.prefix_hit_rate
        else bounds.sf_ttft
    )
    eff_isl_c = max(1, math.ceil((1.0 - h_eff) * eval_context.scenario.isl))
    n_chunks_c = max(
        1, math.ceil(eff_isl_c / max(1, eval_context.scenario.prefill_chunk))
    )
    chunk_tok_c = min(eff_isl_c, eval_context.scenario.prefill_chunk)
    sf_dec = _eval_decode_floor(eval_context, topo, par, b_res)
    sf_ver_only = (
        step_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            StepShape(
                batch=b_res,
                rows_per_seq=eval_context.spec_k + 1,
                context=eval_context.ctx_mid,
                state_writes=1,
            ),
            FloorOptions(dt=eval_context.dtypes, topology=topo),
        )
        if eval_context.spec_k > 0
        else sf_dec
    )
    draft_serial_s = max(0.0, sf_dec.serial_estimate_s - sf_ver_only.serial_estimate_s)
    sf_mix = step_floor(
        eval_context.geom,
        eval_context.chip,
        par,
        StepShape(
            batch=b_res,
            rows_per_seq=eval_context.spec_k + 1,
            context=eval_context.ctx_mid,
            state_writes=1,
            prefill_tokens=chunk_tok_c,
            prefix_hit_rate=0.0,
        ),
        FloorOptions(dt=eval_context.dtypes, topology=topo),
    )
    mix_serial_s = sf_mix.serial_estimate_s + draft_serial_s
    total_req_work_s = (
        dec_steps_per_req * sf_dec.serial_estimate_s
        + float(b_res * n_chunks_c) * mix_serial_s
    )
    res_cap = float(b_res * max(1, par.dp))
    c_served = min(float(c), res_cap)
    server_cap_rate = res_cap / max(1e-9, total_req_work_s)
    w_srv_final = (
        sf_ttft_c.serial_estimate_s + dec_steps_per_req * sf_dec.serial_estimate_s
    )
    rate_req_s = (
        min(
            server_cap_rate,
            float(c) / max(1e-9, w_srv_final + eval_context.scenario.think_time_s),
        )
        if eval_context.scenario.think_time_s > 0.0
        else c_served / max(1e-9, total_req_work_s)
    )
    p_mix = min(
        1.0,
        float(b_res * n_chunks_c)
        / float(max(1.0, dec_steps_per_req + float(b_res * n_chunks_c))),
    )
    p90_step_s = (1.0 - p_mix) * sf_dec.serial_estimate_s + p_mix * mix_serial_s
    p90_tpot_ms = p90_step_s * UNIT_MILLIS_PER_SECOND / eval_context.eff_al
    c_demand = min(float(c), float(b_local * max(1, par.dp)))
    queue_wait_s = max(0.0, (c_demand / max(1e-9, res_cap)) - 1.0) * w_srv_final
    ttft_base_ms = (sf_ttft_c.serial_estimate_s + queue_wait_s) * UNIT_MILLIS_PER_SECOND
    p90_ttft_ms = ttft_base_ms + (
        sf_mix.serial_estimate_s * UNIT_MILLIS_PER_SECOND if c > 1 else 0.0
    )
    interactivity = UNIT_MILLIS_PER_SECOND / max(1e-6, p90_tpot_ms)
    out_tput = rate_req_s * float(eval_context.scenario.osl) / float(topo.num_chips)
    tot_tput = (
        rate_req_s
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
        / float(topo.num_chips)
    )
    slo_ok = (
        eval_context.scenario.ttft_slo_ms is None
        or p90_ttft_ms <= eval_context.scenario.ttft_slo_ms
    ) and (
        eval_context.scenario.tpot_slo_ms is None
        or p90_tpot_ms <= eval_context.scenario.tpot_slo_ms
    )
    max_seqs_flag = b_res if b_res < b_local else eval_context.scenario.max_num_seqs
    pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="mixed",
            instances=1,
            topo=topo,
            par=par,
            hbm_used_gb=hbm_b / UNIT_GIGA,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            max_num_seqs=max_seqs_flag,
            max_model_len=eval_context.ctx_full,
            hbm_weights_gb=hbm_bd.weights / UNIT_GIGA,
            hbm_kv_gb=hbm_bd.kv / UNIT_GIGA,
            hbm_state_gb=hbm_bd.state / UNIT_GIGA,
        )
    )
    binding = (
        f"dec:{sf_dec.binding},mix:{sf_mix.binding}" if p_mix > 0.0 else sf_dec.binding
    )
    return ParetoPoint(
        mode="mixed",
        concurrency=c,
        total_chips=topo.num_chips,
        prefill_pool=None,
        decode_pool=pool,
        p90_ttft_ms=p90_ttft_ms,
        p90_tpot_ms=p90_tpot_ms,
        p90_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=p_mix,
        binding=binding,
        slo_met=slo_ok,
        decode_floor=_floor_times(sf_dec),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=h_eff,
    )


def _decode_capacity_profile(
    eval_context: _SolveContext,
    par: Parallel,
) -> tuple[float, int, int, float, float]:
    """Compute (w_base_b, b_hbm_max, b_smem_max, b_kv_phys_core, hbm_c1)."""
    w_base_b = hbm_footprint_per_core_bytes(
        eval_context.geom, eval_context.dtypes, par, SeqKvLayout(0, 0)
    )
    hbm_c1 = hbm_footprint_per_core_bytes(
        eval_context.geom,
        eval_context.dtypes,
        par,
        SeqKvLayout(1, eval_context.ctx_full),
    )
    per_seq_b = hbm_c1 - w_base_b
    b_hbm_max = (
        int((eval_context.hbm_limit_b - w_base_b) // per_seq_b)
        if (hbm_c1 <= eval_context.hbm_limit_b and per_seq_b > 0.0)
        else (eval_context.c_max if hbm_c1 <= eval_context.hbm_limit_b else 0)
    )
    b_smem_max = smem_max_seqs(
        eval_context.ctx_full, par.kv_split, eval_context.page_size
    )
    b_kv_phys_core = hbm_footprint_per_core_bytes(
        eval_context.geom, eval_context.dtypes, par, SeqKvLayout(1, 1)
    ) - hbm_footprint_per_core_bytes(
        eval_context.geom, eval_context.dtypes, par, SeqKvLayout(1, 0)
    )
    return w_base_b, b_hbm_max, b_smem_max, b_kv_phys_core, hbm_c1


def _evaluate_mixed_candidates(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[ParetoPoint], float, float]:
    """Evaluate all single-pool mixed candidate parallelisms across topologies."""
    candidates: list[ParetoPoint] = []
    min_c1_tpot_ms = math.inf
    min_req_ttft_ms = math.inf
    for topo in topos:
        for par in _candidate_decode_parallels(eval_context.geom, topo):
            w_base_b, b_hbm_max, b_smem_max, b_kv_phys_core, hbm_c1 = (
                _decode_capacity_profile(eval_context, par)
            )
            if hbm_c1 <= eval_context.hbm_limit_b and b_smem_max >= 1:
                sf_c1 = _eval_decode_floor(eval_context, topo, par, 1)
                min_c1_tpot_ms = min(
                    min_c1_tpot_ms,
                    sf_c1.serial_estimate_s
                    * UNIT_MILLIS_PER_SECOND
                    / eval_context.eff_al,
                )
            if eval_context.mode not in ("auto", "mixed"):
                continue
            sf_ttft = _prefill_step_floor(
                eval_context, topo, par, eval_context.scenario.prefix_hit_rate
            )
            min_req_ttft_ms = min(
                min_req_ttft_ms, sf_ttft.serial_estimate_s * UNIT_MILLIS_PER_SECOND
            )
            bounds = _MixedPointBounds(
                w_base_b=w_base_b,
                b_kv_phys_core=b_kv_phys_core,
                b_hbm_max=b_hbm_max,
                b_smem_max=b_smem_max,
                sf_ttft=sf_ttft,
            )
            for c in eval_context.scenario.concurrencies:
                eval_context.enumerated += 1
                pt = _evaluate_mixed_point(eval_context, topo, par, c, bounds)
                if pt is not None:
                    candidates.append(pt)
    return candidates, min_c1_tpot_ms, min_req_ttft_ms


def _compute_disagg_metrics(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    target: _DisaggRatioTarget,
    batch_stats: _DisaggBatchStats,
    sf_d: StepFloor,
) -> tuple[float, float, float, float, float]:
    """Compute serving latencies and throughputs for disaggregated configuration."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    p90_tpot_ms = sf_d.serial_estimate_s * UNIT_MILLIS_PER_SECOND / eval_context.eff_al
    interactivity = UNIT_MILLIS_PER_SECOND / max(1e-6, p90_tpot_ms)
    res_cap_d = float(batch_stats.b_res * max(1, target.y_d * dec.par.dp))
    c_served_d = min(float(target.c), res_cap_d)
    rate_d_cap = c_served_d / (dec_steps_per_req * max(1e-9, sf_d.serial_estimate_s))
    c_demand_d = min(
        float(target.c), float(batch_stats.b_local * max(1, target.y_d * dec.par.dp))
    )
    queue_wait_d_ms = (
        max(0.0, (c_demand_d / max(1e-9, res_cap_d)) - 1.0)
        * (dec_steps_per_req * sf_d.serial_estimate_s)
        * UNIT_MILLIS_PER_SECOND
    )
    p90_ttft_d_ms = batch_stats.ttft_eff_ms + queue_wait_d_ms
    if eval_context.scenario.think_time_s > 0.0:
        client_rate = float(target.c) / max(
            1e-9,
            batch_stats.ttft_eff_ms * UNIT_MICROS_PER_SECOND * UNIT_MILLIS_PER_SECOND
            + dec_steps_per_req * sf_d.serial_estimate_s
            + eval_context.scenario.think_time_s,
        )
        coupled_rate = min(target.x_p * batch_stats.rate_p_eff, rate_d_cap, client_rate)
    else:
        coupled_rate = min(target.x_p * batch_stats.rate_p_eff, rate_d_cap)
    out_tput = (
        coupled_rate * float(eval_context.scenario.osl) / float(target.total_chips)
    )
    tot_tput = (
        coupled_rate
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
        / float(target.total_chips)
    )
    return p90_ttft_d_ms, p90_tpot_ms, interactivity, out_tput, tot_tput


def _solve_disagg_think_batch(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
) -> int:
    """Resolve Little's-law active local batch for disaggregated decode pool."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    b_local = max(1, math.ceil(target.c / max(1, target.y_d * dec.par.dp)))
    if eval_context.scenario.think_time_s <= 0.0:
        return b_local
    for _ in range(2):
        sf_d_tmp = _eval_decode_floor(eval_context, dec.topo, dec.par, b_local)
        w_srv = (
            pref.ttft_ms / UNIT_MILLIS_PER_SECOND
            + dec_steps_per_req * sf_d_tmp.serial_estimate_s
        )
        c_act = float(target.c) * w_srv / (w_srv + eval_context.scenario.think_time_s)
        b_local = max(1, math.ceil(c_act / max(1, target.y_d * dec.par.dp)))
    return b_local


def _resolve_effective_prefill(
    eval_context: _SolveContext,
    pref: _PrefillOption,
    h_eff: float,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> tuple[float, float]:
    """Recompute cached prefill (ttft_ms, rate_p) when prefix hit rate is clamped."""
    if h_eff >= eval_context.scenario.prefix_hit_rate:
        return pref.ttft_ms, pref.rate_p
    p_key = (pref.topo.name, pref.par, round(h_eff, 6))
    cached_p = cache.get(p_key)
    if cached_p is not None:
        return cached_p
    eff_isl_p = max(1, math.ceil((1.0 - h_eff) * eval_context.scenario.isl))
    n_chunks_p = max(
        1, math.ceil(eff_isl_p / max(1, eval_context.scenario.prefill_chunk))
    )
    sf_p_eff = _prefill_step_floor(eval_context, pref.topo, pref.par, h_eff)
    ttft_p_eff = (
        sf_p_eff.serial_estimate_s
        * (1.0 + float(pref.par.pp - 1) / float(max(1, n_chunks_p)))
        * UNIT_MILLIS_PER_SECOND
    )
    rate_p_eff = 1.0 / max(1e-9, sf_p_eff.serial_estimate_s)
    cached_p = (ttft_p_eff, rate_p_eff)
    cache[p_key] = cached_p
    return cached_p


def _evaluate_disagg_instance_pair(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> ParetoPoint | None:
    """Evaluate one instance ratio pair (x_p, y_d) for disaggregated serving."""
    if (
        eval_context.max_chips is not None
        and target.total_chips > eval_context.max_chips
    ):
        return None
    if dec.b_hbm_max < 1 or dec.b_smem_max < 1:
        return None
    b_local = _solve_disagg_think_batch(eval_context, dec, pref, target)
    b_start = min(b_local, dec.b_hbm_max, dec.b_smem_max)
    if eval_context.scenario.max_num_seqs is not None:
        b_start = min(b_start, eval_context.scenario.max_num_seqs)
    clamped = _clamp_resident_batch(eval_context, dec.par, b_start)
    if clamped is None:
        return None
    b_res, hbm_bd_d = clamped
    hbm_d = hbm_bd_d.total
    eval_context.hbm_fit += 1
    cap_tok, h_eff = kv_capacity_and_hit_rate(
        eval_context.geom,
        eval_context.dtypes,
        dec.par,
        b_res,
        KvFitInput(
            b_ws=b_local,
            ctx_full=eval_context.ctx_full,
            hbm_limit_b=eval_context.hbm_limit_b,
            w_base_b=dec.w_base_b,
            b_kv_phys_core=dec.b_kv_phys,
            offload_b_per_core=dec.offload_b_per_core,
            prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
            kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
        ),
    )
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    ttft_eff_ms, rate_p_eff = _resolve_effective_prefill(
        eval_context, pref, h_eff, cache
    )
    sf_d = _eval_decode_floor(eval_context, dec.topo, dec.par, b_res)
    p90_ttft_d_ms, p90_tpot_ms, interactivity, out_tput, tot_tput = (
        _compute_disagg_metrics(
            eval_context,
            dec,
            target,
            _DisaggBatchStats(b_res, b_local, ttft_eff_ms, rate_p_eff),
            sf_d,
        )
    )
    slo_ok = (
        eval_context.scenario.ttft_slo_ms is None
        or p90_ttft_d_ms <= eval_context.scenario.ttft_slo_ms
    ) and (
        eval_context.scenario.tpot_slo_ms is None
        or p90_tpot_ms <= eval_context.scenario.tpot_slo_ms
    )
    max_seqs_d = b_res if b_res < b_local else eval_context.scenario.max_num_seqs
    p_pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="prefill",
            instances=target.x_p,
            topo=pref.topo,
            par=pref.par,
            hbm_used_gb=pref.hbm_gb,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            max_num_seqs=None,
            max_model_len=eval_context.ctx_full,
            hbm_weights_gb=pref.hbm_bd.weights / UNIT_GIGA,
            hbm_kv_gb=pref.hbm_bd.kv / UNIT_GIGA,
            hbm_state_gb=pref.hbm_bd.state / UNIT_GIGA,
        )
    )
    d_pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="decode",
            instances=target.y_d,
            topo=dec.topo,
            par=dec.par,
            hbm_used_gb=hbm_d / UNIT_GIGA,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            max_num_seqs=max_seqs_d,
            max_model_len=eval_context.ctx_full,
            hbm_weights_gb=hbm_bd_d.weights / UNIT_GIGA,
            hbm_kv_gb=hbm_bd_d.kv / UNIT_GIGA,
            hbm_state_gb=hbm_bd_d.state / UNIT_GIGA,
        )
    )
    return ParetoPoint(
        mode="disagg",
        concurrency=target.c,
        total_chips=target.total_chips,
        prefill_pool=p_pool,
        decode_pool=d_pool,
        p90_ttft_ms=p90_ttft_d_ms,
        p90_tpot_ms=p90_tpot_ms,
        p90_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=0.0,
        binding=sf_d.binding,
        slo_met=slo_ok,
        decode_floor=_floor_times(sf_d),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=h_eff,
    )


def _search_disagg_instance_ratios(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    c: int,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> list[ParetoPoint]:
    """Search instance multiplier ratios (x_p, y_d) for one prefill-decode pair."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    b_one = math.ceil(c / max(1, dec.par.dp))
    sf_d_one = _eval_decode_floor(eval_context, dec.topo, dec.par, b_one)
    rate_d_one = float(c) / (dec_steps_per_req * max(1e-9, sf_d_one.serial_estimate_s))
    eff_max_inst = (
        eval_context.max_instances
        if eval_context.max_instances is not None
        else DEFAULT_MAX_INSTANCES
    )
    if eval_context.max_chips is None:
        ratio_xp = math.ceil(rate_d_one / max(1e-9, pref.rate_p)) + 1
        ratio_yd = math.ceil(pref.rate_p / max(1e-9, rate_d_one)) + 1
        max_xp = min(eff_max_inst, max(1, ratio_xp))
        max_yd = min(eff_max_inst, max(1, ratio_yd))
    else:
        max_xp = min(
            eff_max_inst, max(1, eval_context.max_chips // pref.topo.num_chips)
        )
        max_yd = min(eff_max_inst, max(1, eval_context.max_chips // dec.topo.num_chips))
    points: list[ParetoPoint] = []
    for x_p in range(1, max_xp + 1):
        for y_d in range(1, max_yd + 1):
            total_chips = x_p * pref.topo.num_chips + y_d * dec.topo.num_chips
            if (
                eval_context.max_chips is not None
                and total_chips > eval_context.max_chips
            ):
                continue
            eval_context.enumerated += 1
            pt = _evaluate_disagg_instance_pair(
                eval_context,
                dec,
                pref,
                _DisaggRatioTarget(c=c, x_p=x_p, y_d=y_d, total_chips=total_chips),
                cache,
            )
            if pt is not None:
                points.append(pt)
    return points


def _evaluate_disaggregated_candidates(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
    prefill_options: list[_PrefillOption],
) -> list[ParetoPoint]:
    """Evaluate all unconstrained disaggregated serving candidate configurations."""
    if eval_context.mode not in ("auto", "disagg") or not prefill_options:
        return []
    candidates: list[ParetoPoint] = []
    cache: dict[tuple[str, Parallel, float], tuple[float, float]] = {}
    for topo_d in topos:
        offload_b = (
            float(eval_context.scenario.kv_offload_gb)
            * UNIT_GIGA
            / float(max(1, topo_d.num_cores))
        )
        for par_d in _candidate_decode_parallels(eval_context.geom, topo_d):
            w_base_d, b_hbm_max_d, b_smem_max_d, b_kv_phys_d, _ = (
                _decode_capacity_profile(eval_context, par_d)
            )
            dec_state = _DisaggDecodeState(
                topo=topo_d,
                par=par_d,
                w_base_b=w_base_d,
                b_hbm_max=b_hbm_max_d,
                b_smem_max=b_smem_max_d,
                b_kv_phys=b_kv_phys_d,
                offload_b_per_core=offload_b,
            )
            for c in eval_context.scenario.concurrencies:
                for pref in prefill_options:
                    if not _valid_raiden_gdn_pair(
                        eval_context.geom, pref.par, par_d
                    ) or not _valid_raiden_kv_pair(eval_context.geom, pref.par, par_d):
                        continue
                    candidates.extend(
                        _search_disagg_instance_ratios(
                            eval_context, dec_state, pref, c, cache
                        ),
                    )
    return candidates


def _resolve_solve_topologies(opts: SolveOptions) -> list[SliceTopology]:
    """Resolve provisioned topologies within max_chips budget, rejecting subhost."""
    topologies = (
        list(opts.topologies)
        if opts.topologies is not None
        else list(TOPOLOGIES.values())
    )
    if opts.max_chips is not None:
        if opts.max_chips < MIN_SLICE_CHIPS and opts.topologies is None:
            min_topo = next(iter(TOPOLOGIES))
            msg = (
                f"max_chips={opts.max_chips} is below minimum provisionable TPU "
                f"topology {min_topo!r} ({MIN_SLICE_CHIPS} chips)"
            )
            raise ValueError(msg)
        return [t for t in topologies if t.num_chips <= opts.max_chips]
    return topologies


def solve_serving_plan(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    options: SolveOptions | None = None,
) -> ServingPlan:
    """Search single-pool mixed and unconstrained xP/yD disagg plans."""
    opts = options or SolveOptions()
    if not (0.0 < opts.gpu_memory_utilization <= 1.0):
        msg = (
            f"gpu_memory_utilization must be in (0, 1], got "
            f"{opts.gpu_memory_utilization}"
        )
        raise ValueError(msg)
    topos = _resolve_solve_topologies(opts)
    hbm_limit_b = float(opts.gpu_memory_utilization) * float(chip.hbm_bytes)
    eff_al = max(1.0, scenario.accept_len)
    spec_k = (
        scenario.spec_tokens
        if scenario.spec_tokens is not None
        else (max(geom.mtp_layers, math.ceil(eff_al) - 1) if eff_al > 1.0 else 0)
    )
    ctx_full = scenario.isl + scenario.osl
    eff_isl = max(1, math.ceil((1.0 - scenario.prefix_hit_rate) * scenario.isl))
    n_chunks = max(1, math.ceil(eff_isl / max(1, scenario.prefill_chunk)))
    page_size = (
        scenario.page_size
        if scenario.page_size is not None
        else default_rpa_page_size(ctx_full)
    )

    eval_context = _SolveContext(
        geom=geom,
        dtypes=dtypes,
        chip=chip,
        scenario=scenario,
        mode=opts.mode,
        max_chips=opts.max_chips,
        max_instances=opts.max_instances,
        gpu_memory_utilization=opts.gpu_memory_utilization,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / UNIT_GIGA,
        eff_al=eff_al,
        spec_k=spec_k,
        ctx_full=ctx_full,
        ctx_mid=scenario.isl + scenario.osl // 2,
        eff_isl=eff_isl,
        n_chunks=n_chunks,
        page_size=page_size,
        c_max=max(scenario.concurrencies),
    )

    prefill_options, min_prefill_ttft_ms = _find_prefill_options(eval_context, topos)
    mixed_candidates, min_c1_tpot_ms, min_mixed_ttft_ms = _evaluate_mixed_candidates(
        eval_context,
        topos,
    )
    disagg_candidates = _evaluate_disaggregated_candidates(
        eval_context, topos, prefill_options
    )
    candidates = mixed_candidates + disagg_candidates

    warnings: list[str] = []
    min_req_ttft_ms = min(min_prefill_ttft_ms, min_mixed_ttft_ms)
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

    slo_candidates = [p for p in candidates if p.slo_met]
    pool_for_frontier = slo_candidates if slo_candidates else candidates
    if candidates and not slo_candidates and not warnings:
        warnings.append("SLO_UNMET: no feasible topology met both TTFT and TPOT bounds")

    frontier = _pareto_filter(pool_for_frontier)
    recommended = (
        max(frontier, key=lambda p: p.token_throughput_per_chip) if frontier else None
    )
    return ServingPlan(
        model_layers=geom.num_layers,
        chip=chip.name,
        scenario=scenario,
        mode=opts.mode,
        recommended=recommended,
        frontier=frontier,
        warnings=tuple(warnings),
        candidates=tuple(pool_for_frontier),
        gpu_memory_utilization=opts.gpu_memory_utilization,
        search=SearchStats(
            enumerated=eval_context.enumerated,
            hbm_fit=eval_context.hbm_fit,
            slo_met=len(slo_candidates),
            frontier=len(frontier),
        ),
    )
