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
"""Megakernel advisor data schemas, constants, and workload specifications."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict

from ..core import errors
from . import chips, constants
from .constants import EFFICIENCY_BRACKETS, MEASUREMENTS_SCHEMA
from .models import ModelProfile, ParallelLayout

if TYPE_CHECKING:
    from ..core.exec_spec import Tiles
    from .chips import ChipSpec

DTYPE_BYTES = constants.DTYPE_BYTES

FusionScope = Literal["per_op", "single_layer", "layer_group", "whole_step"]
Phase = Literal["decode", "verify", "prefill", "mixed"]
# Where routed tokens are grouped by expert and where the data lives.
MoEMode = Literal["masked_on_chip", "grouped_on_chip", "grouped_in_hbm", "dense_mlp"]
ProvenanceTag = Literal["DERIVED", "ASSUMED", "MEASURED"]
ObjectiveMode = Literal["step", "throughput", "both"]
DispatchRegime = Literal["auto", "zero_floor", "compiled_xla", "unfused_eager"]
SupportStatus = Literal["baseline", "within_core_limits", "declared", "hypothetical"]

SUPPORTED_ONCHIP_FAMILIES: frozenset[str] = frozenset(
    {"gqa", "mla", "gdn", "kda", "dense_mlp", "moe"}
)


def _scalar_float(val: object, field_name: str) -> float:
    """Parse a finite JSON int/float scalar, rejecting bools/strings."""
    match val:
        case bool():
            msg = f"{field_name} must be numeric, got bool"
            raise errors.MegakernelConfigError(msg)
        case int() | float():
            num = float(val)
            if not math.isfinite(num):
                msg = f"{field_name} must be finite, got {val!r}"
                raise errors.MegakernelConfigError(msg)
            return num
        case _:
            msg = f"{field_name} must be numeric, got {type(val).__name__}"
            raise errors.MegakernelConfigError(msg)


def _scalar_int(val: object, field_name: str) -> int:
    """Parse an integer scalar cleanly, rejecting bools and non-integers."""
    match val:
        case bool():
            msg = f"{field_name} must be an integer, got bool"
            raise errors.MegakernelConfigError(msg)
        case int():
            return val
        case _:
            msg = f"{field_name} must be an integer, got {type(val).__name__}"
            raise errors.MegakernelConfigError(msg)


class ProvenanceMetric(TypedDict):
    """Single output metric value paired with its provenance tag, source, and band."""

    value: float | int | bool
    unit: str
    tag: ProvenanceTag
    source: str
    band: list[float]


@dataclass(frozen=True)
class RingPlan:
    """On-chip VMEM ring-buffer slot counts and bank widths for weight streaming."""

    dense_slots: int = 2
    expert_slots: int = 2
    dense_slot_bytes: int = 2 << 20
    expert_slot_bytes: int = 1 << 20
    compiler_spill_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        """Validate positive VMEM ring slot counts and byte widths."""
        if self.dense_slots < 1:
            msg = f"RingPlan.dense_slots must be >= 1, got {self.dense_slots}"
            raise errors.MegakernelConfigError(msg)
        if self.expert_slots < 0:
            msg = f"RingPlan.expert_slots must be >= 0, got {self.expert_slots}"
            raise errors.MegakernelConfigError(msg)
        if self.dense_slot_bytes <= 0:
            msg = f"RingPlan.dense_slot_bytes must be > 0, got {self.dense_slot_bytes}"
            raise errors.MegakernelConfigError(msg)
        if self.expert_slot_bytes < 0 or self.compiler_spill_reserve_bytes < 0:
            msg = (
                "RingPlan.expert_slot_bytes and compiler_spill_reserve_bytes "
                "must be >= 0"
            )
            raise errors.MegakernelConfigError(msg)


def default_double_buffer_ring_plan(dma_buffers: int = 2) -> RingPlan:
    """Return the default n-slot weight-streaming RingPlan."""
    return RingPlan(dense_slots=dma_buffers, expert_slots=dma_buffers)


@dataclass(frozen=True)
class MeasurementsData:
    """Calibrated runtime measurements loaded from an `advisor.measurements.v1` file."""

    source_desc: str
    source_kind: str = "trace"
    launch_gap_p50_us: float | None = None
    launch_gap_p90_us: float | None = None
    launch_gap_n: int = 0
    collective_hop_us: float | None = None
    collective_n: int = 0
    step_coverage: float | None = None
    step_time_us: float | None = None
    dcn_bw_gbps: float | None = None
    ring_plan: RingPlan | None = None


def _optional_mapping(raw: object, field_name: str) -> Mapping[str, object]:
    """Return a Mapping or empty dict when None; reject invalid types."""
    match raw:
        case None:
            return {}
        case Mapping() as mapping:
            return mapping
        case _:
            msg = f"{field_name} must be a mapping, got {type(raw).__name__}"
            raise errors.MegakernelConfigError(msg)


def _scalar_str(val: object, field_name: str) -> str:
    """Require an exact string scalar."""
    if type(val) is not str:
        msg = f"{field_name} must be a string, got {type(val).__name__}"
        raise errors.MegakernelConfigError(msg)
    return val


def _parse_collectives_hop_us(coll_raw: object) -> tuple[float | None, int]:
    """Extract minimum per-hop collective latency and count from collectives."""
    if coll_raw is None:
        return None, 0
    if type(coll_raw) is not list:
        msg = f"collectives must be a list, got {type(coll_raw).__name__}"
        raise errors.MegakernelConfigError(msg)
    if not coll_raw:
        return None, 0
    hops_list: list[float] = []
    coll_n = 0
    for idx, item in enumerate(coll_raw):
        match item:
            case Mapping() as entry:
                if "latency_us" not in entry:
                    continue
                latency_us = _scalar_float(
                    entry["latency_us"], f"collectives[{idx}].latency_us"
                )
                if latency_us < 0.0:
                    msg = f"collectives[{idx}].latency_us must be >= 0"
                    raise errors.MegakernelConfigError(msg)
                grp_raw = entry.get("group_size", 2)
                grp = max(2, _scalar_int(grp_raw, f"collectives[{idx}].group_size"))
                n_raw = entry.get("n", 1)
                n_val = _scalar_int(n_raw, f"collectives[{idx}].n")
                if n_val < 0:
                    msg = f"collectives[{idx}].n must be >= 0"
                    raise errors.MegakernelConfigError(msg)
                hops = max(1, round(log2(grp)))
                hops_list.append(latency_us / hops)
                coll_n += n_val
            case _:
                msg = f"collectives[{idx}] must be a mapping"
                raise errors.MegakernelConfigError(msg)
    return (min(hops_list) if hops_list else None), coll_n


def _parse_ring_plan(ring_plan_raw: object) -> RingPlan | None:
    """Parse an optional RingPlan mapping from measurements JSON."""
    if ring_plan_raw is None:
        return None
    ring_plan_map = _optional_mapping(ring_plan_raw, "ring_plan")
    return RingPlan(
        dense_slots=_scalar_int(
            ring_plan_map.get("dense_slots", 2), "ring_plan.dense_slots"
        ),
        expert_slots=_scalar_int(
            ring_plan_map.get("expert_slots", 2), "ring_plan.expert_slots"
        ),
        dense_slot_bytes=_scalar_int(
            ring_plan_map.get("dense_slot_bytes", 2 << 20), "ring_plan.dense_slot_bytes"
        ),
        expert_slot_bytes=_scalar_int(
            ring_plan_map.get("expert_slot_bytes", 1 << 20),
            "ring_plan.expert_slot_bytes",
        ),
        compiler_spill_reserve_bytes=_scalar_int(
            ring_plan_map.get("compiler_spill_reserve_bytes", 0),
            "ring_plan.compiler_spill_reserve_bytes",
        ),
    )


def _load_measurements_payload(
    target: object, chip_name: str
) -> tuple[Mapping[str, object], str]:
    """Resolve and validate the top-level measurements JSON mapping."""
    match target:
        case Mapping() as mapping:
            return mapping, "inline_mapping"
        case str() | Path():
            target_str = str(target).strip()
            if not target_str:
                msg = "Measurements target path must not be empty"
                raise errors.MegakernelConfigError(msg)
            if target_str in ("bundled", "auto"):
                bundled_path = chips.public_calibration_path(chip_name)
                if bundled_path is None or not bundled_path.is_file():
                    msg = (
                        "No default bundled architectural priors found for chip "
                        f"{chip_name!r}"
                    )
                    raise errors.MegakernelConfigError(msg)
                path = bundled_path
            else:
                path = Path(target_str)
                if not path.is_file():
                    msg = f"Missing measurements file: {path}"
                    raise errors.MegakernelConfigError(msg)
            try:
                parsed: object = json.loads(path.read_text("utf-8"))
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in measurements file {path}: {exc}"
                raise errors.MegakernelConfigError(msg) from exc
            match parsed:
                case Mapping() as mapping:
                    return mapping, path.name
                case _:
                    msg = f"Measurements file {path} must contain a JSON object"
                    raise errors.MegakernelConfigError(msg)
        case _:
            msg = (
                "Measurements target must be a path or mapping, "
                f"got {type(target).__name__}"
            )
            raise errors.MegakernelConfigError(msg)


def _parse_launch_gap(
    payload: Mapping[str, object],
) -> tuple[float | None, float | None, int]:
    """Parse optional launch_gap_us statistics, failing closed on bad scalars."""
    lg_map = _optional_mapping(payload.get("launch_gap_us"), "launch_gap_us")
    if "p50" not in lg_map:
        return None, None, 0
    lg_p50 = _scalar_float(lg_map["p50"], "launch_gap_us.p50")
    if lg_p50 < 0.0:
        msg = f"launch_gap_us.p50 must be >= 0, got {lg_p50}"
        raise errors.MegakernelConfigError(msg)
    lg_p90 = _scalar_float(lg_map.get("p90", lg_p50), "launch_gap_us.p90")
    if lg_p90 < 0.0:
        msg = f"launch_gap_us.p90 must be >= 0, got {lg_p90}"
        raise errors.MegakernelConfigError(msg)
    lg_n = _scalar_int(lg_map.get("n", 0), "launch_gap_us.n")
    if lg_n < 0:
        msg = f"launch_gap_us.n must be >= 0, got {lg_n}"
        raise errors.MegakernelConfigError(msg)
    return lg_p50, lg_p90, lg_n


def _parse_step_time(st_raw: object) -> float | None:
    """Parse optional step_time_us scalar or `{p50: ...}` mapping."""
    match st_raw:
        case None:
            return None
        case Mapping() as st_map:
            if "p50" not in st_map:
                return None
            st_val = _scalar_float(st_map["p50"], "step_time_us.p50")
        case _:
            st_val = _scalar_float(st_raw, "step_time_us")
    if st_val < 0.0:
        msg = f"step_time_us must be >= 0, got {st_val}"
        raise errors.MegakernelConfigError(msg)
    return st_val


def load_measurements(
    target: object,
    *,
    chip_name: str = "tpu7x",
) -> MeasurementsData:
    """Load and validate an `advisor.measurements.v1` JSON file or mapping."""
    if type(target) is MeasurementsData:
        return target
    payload, file_label = _load_measurements_payload(target, chip_name)
    if payload.get("schema") != MEASUREMENTS_SCHEMA:
        msg = (
            f"Unsupported measurements schema {payload.get('schema')!r}; "
            f"expected {MEASUREMENTS_SCHEMA!r}"
        )
        raise errors.MegakernelConfigError(msg)
    src_map = _optional_mapping(payload.get("source"), "source")
    src_kind = _scalar_str(src_map.get("kind", "trace"), "source.kind")
    src_file = _scalar_str(src_map.get("file", file_label), "source.file")
    src_desc = f"{src_kind}:{src_file}"

    lg_p50, lg_p90, lg_n = _parse_launch_gap(payload)
    coll_hop, coll_n = _parse_collectives_hop_us(payload.get("collectives"))

    cov_map = _optional_mapping(payload.get("step_coverage"), "step_coverage")
    cov_val = (
        _scalar_float(
            cov_map["fused_eligible_fraction"],
            "step_coverage.fused_eligible_fraction",
        )
        if "fused_eligible_fraction" in cov_map
        else None
    )
    if cov_val is not None and not (0.0 <= cov_val <= 1.0):
        msg = f"step_coverage.fused_eligible_fraction must be in [0, 1], got {cov_val}"
        raise errors.MegakernelConfigError(msg)

    st_val = _parse_step_time(payload.get("step_time_us"))

    dcn_raw = payload.get("dcn_bw_gbps")
    dcn_val = _scalar_float(dcn_raw, "dcn_bw_gbps") if dcn_raw is not None else None
    if dcn_val is not None and dcn_val <= 0.0:
        msg = f"dcn_bw_gbps must be > 0, got {dcn_val}"
        raise errors.MegakernelConfigError(msg)

    return MeasurementsData(
        source_desc=src_desc,
        source_kind=src_kind,
        launch_gap_p50_us=lg_p50,
        launch_gap_p90_us=lg_p90,
        launch_gap_n=lg_n,
        collective_hop_us=coll_hop,
        collective_n=coll_n,
        step_coverage=cov_val,
        step_time_us=st_val,
        dcn_bw_gbps=dcn_val,
        ring_plan=_parse_ring_plan(payload.get("ring_plan")),
    )


@dataclass(frozen=True)
class EfficiencySpec:
    """ASSUMED achievable unit utilization priors and brackets vs peak rates."""

    BRACKETS: ClassVar[Mapping[str, tuple[float, float]]] = EFFICIENCY_BRACKETS

    eta_hbm: float = 0.82
    eta_dense: float = 0.72
    eta_moe_grouped: float = 0.50
    eta_ici: float = 0.75

    def __post_init__(self) -> None:
        """Validate all utilization efficiencies lie strictly in (0, 1]."""
        for field_name, val in (
            ("eta_hbm", self.eta_hbm),
            ("eta_dense", self.eta_dense),
            ("eta_moe_grouped", self.eta_moe_grouped),
            ("eta_ici", self.eta_ici),
        ):
            if not math.isfinite(val) or not (0.0 < val <= 1.0):
                msg = f"EfficiencySpec.{field_name} must be in (0, 1], got {val}"
                raise errors.MegakernelConfigError(msg)


@dataclass(frozen=True)
class WorkloadSpec:
    """Serving request geometry, VMEM budget, and multi-buffering configuration."""

    phase: Phase = "decode"
    batch: int = 16
    isl: int = 4096
    osl: int = 1024
    context: int = 0
    prefix_hit_rate: float = 0.0
    spec_tokens: int = 0
    chunk_size: int = 0
    microbatches: int = 0
    tp: int = 8
    pp: int = 1
    dp: int = 1
    ep: int = 1
    kv_split: int = 1
    dcp: int = 1
    pcp: int = 1
    segment_layers: int = 0
    # Layers fused per kernel call for the layer_group scope, in call order
    # (empty: the model has no shipped layer grouping). Must sum to num_layers.
    layer_groups: tuple[int, ...] = ()
    moe_mode: MoEMode = "masked_on_chip"
    vmem_alias: bool = True
    vmem_fit_headroom: bool = False
    tiles: Tiles | None = None
    op_overhead_us: float = 0.0
    vmem_budget_mib: float | None = None
    dma_buffers: int = 2
    ring_plan: RingPlan | None = None
    objective: ObjectiveMode = "step"
    hbm_utilization_cap: float = 0.80
    dispatch_regime: DispatchRegime = "auto"
    efficiency: EfficiencySpec = EfficiencySpec()
    # Most rows the megakernel serves in one decode or verify call. A larger
    # batch runs as separate calls that each stream their weights again.
    # 0: no declared limit.
    max_rows_per_call: int = 0

    def __post_init__(self) -> None:
        """Enforce non-negative finite scalar lengths and overhead inputs."""
        for field_name, val in (
            ("isl", self.isl),
            ("osl", self.osl),
            ("context", self.context),
            ("spec_tokens", self.spec_tokens),
            ("chunk_size", self.chunk_size),
            ("microbatches", self.microbatches),
            ("segment_layers", self.segment_layers),
        ):
            if val < 0:
                msg = f"WorkloadSpec.{field_name} must be >= 0, got {val}"
                raise errors.MegakernelConfigError(msg)
        if not math.isfinite(self.op_overhead_us) or self.op_overhead_us < 0.0:
            msg = (
                "WorkloadSpec.op_overhead_us must be finite and >= 0, "
                f"got {self.op_overhead_us}"
            )
            raise errors.MegakernelConfigError(msg)


@dataclass(frozen=True)
class ScopeEstimate:
    """Roofline and VMEM capacity estimate for one fusion scope."""

    scope: FusionScope
    feasible: bool
    reason: str
    ops_per_step: int
    vmem_static_mib: float
    t_vmem_max: int
    n_splits: int
    w_step_gb: float
    w_crit_gb: float | None
    act_hbm_mb: float
    moe_flop_factor: float
    roof_floor_us: float
    step_bound_us: float
    selected_dense_slots: int = 0
    selected_expert_slots: int = 0
    step_achievable_us: float = 0.0
    break_even_op_overhead_us: float = 0.0
    step_upper_us: float = 0.0
    compute_us: float = 0.0
    hbm_us: float = 0.0
    comm_us: float = 0.0
    fill_drain_us: float = 0.0
    repass_spill_us: float = 0.0
    launch_us: float = 0.0
    step_achievable_upper_us: float = 0.0
    support: SupportStatus = "hypothetical"
    blockers: tuple[str, ...] = ()
    w_step_strict_gb: float = 0.0
    step_floor_strict_us: float = 0.0
    # Dense compute plus the routed-expert stage, max(routed HBM, MoE compute).
    # The core is max(hbm_us, chain_us + comm_us) when fused, else
    # max(hbm_us, chain_us) + comm_us.
    chain_us: float = 0.0

    @property
    def step_ach_upper_us(self) -> float:
        """Alias for step_achievable_upper_us."""
        return self.step_achievable_upper_us

    @property
    def max_vmem_tokens(self) -> int:
        """Return maximum active tokens per step fitting in a single VMEM pass."""
        return self.t_vmem_max

    @property
    def num_vmem_splits(self) -> int:
        """Return the number of VMEM token-dimension passes per step."""
        return self.n_splits


@dataclass(frozen=True)
class AdvisorReport:
    """First-principles fusion-scope recommendation for a single workload."""

    model: str
    chip: str
    phase: Phase
    active_tokens: int
    effective_context: int
    vmem_budget_mib: float
    hbm_fits: bool
    hbm_used_gb: float
    hbm_capacity_gb: float
    ridge_dense_flop_per_byte: float
    step_arithmetic_intensity: float
    asymptotic_decode_intensity: float | None
    recommended_scope: FusionScope
    build_megakernel: bool
    rationale: str
    scopes: tuple[ScopeEstimate, ...]
    per_token_vmem_bytes: int = 0
    ridge_tokens: int = 0
    recommended_moe_mode: str = "dense_mlp"
    dma_buffers: int = 2
    verdict: str = "separate_likely"
    confidence: str = "prior_only"
    flags: tuple[str, ...] = ()
    metrics: dict[str, ProvenanceMetric] = field(default_factory=dict)
    unresolved_assumptions: tuple[dict[str, object], ...] = ()
    fastest_scope: FusionScope = "per_op"
    fastest_verdict: str = "separate_likely"
    evidence: SupportStatus = "baseline"
    hypothetical_best: dict[str, object] | None = None
    coverage: str = (
        "decoder layer stack; excludes embedding, LM head, sampling, "
        "draft heads and modality towers"
    )

    def to_dict(self) -> dict[str, object]:
        """Serialize AdvisorReport to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ParallelismRecommendation:
    """Auto-derived balanced TPU parallelism layout for bare-minimum queries."""

    tp: int
    ep: int
    dp: int
    pp: int
    topology: str
    hbm_used_gb: float
    hbm_capacity_gb: float
    w_step_gb: float
    auto_selected: bool = True
    dcp: int = 1
    pcp: int = 1
    # "deployment preset" when taken from DEPLOYMENT_PRESETS, else empty.
    source: str = ""
    layer_groups: tuple[int, ...] = ()

    def label(self) -> str:
        """Return the layout as `TP8 x EP8 x DCP4`, omitting degree-1 axes."""
        return ParallelLayout(
            tp=self.tp, ep=self.ep, dp=self.dp, pp=self.pp, dcp=self.dcp, pcp=self.pcp
        ).label()


@dataclass(frozen=True)
class RegimeEntry:
    """Evaluation summary for one canonical serving request regime."""

    label: str
    phase: Phase
    active_tokens: int
    effective_context: int
    step_arithmetic_intensity: float
    max_vmem_tokens: int
    num_vmem_splits: int
    recommended_scope: FusionScope
    recommended_moe_mode: str
    step_bound_us: float
    rationale: str
    layout: str = ""
    step_upper_us: float = 0.0
    verdict: str = "separate_likely"
    confidence: str = "prior_only"
    flags: tuple[str, ...] = ()
    metrics: dict[str, ProvenanceMetric] = field(default_factory=dict)
    unresolved_assumptions: tuple[dict[str, object], ...] = ()
    # Floor, upper, launches, and passes of every scope, for the report ledger.
    scopes: tuple[dict[str, object], ...] = ()
    workload: dict[str, object] = field(default_factory=dict)
    fastest_scope: FusionScope = "per_op"
    fastest_verdict: str = "separate_likely"
    evidence: SupportStatus = "baseline"
    step_achievable_us: float = 0.0
    step_achievable_upper_us: float = 0.0
    step_floor_strict_us: float = 0.0
    blockers: tuple[str, ...] = ()
    hypothetical_best: dict[str, object] | None = None

    @property
    def step_ach_upper_us(self) -> float:
        """Alias for step_achievable_upper_us."""
        return self.step_achievable_upper_us


@dataclass(frozen=True)
class DisaggHandoffSpec:
    """Per-request KV and recurrent state wire payload and DCN transfer floor."""

    isl: int
    kv_wire_mb: float
    state_wire_mb: float
    total_wire_mb: float
    dcn_bw_gbps: float
    dcn_bw_gbps_bracket: tuple[float, float]
    dcn_xfer_ms: float
    dcn_xfer_bracket_ms: tuple[float, float]
    dcn_provenance: Literal["ASSUMED", "MEASURED"] = "ASSUMED"


@dataclass(frozen=True)
class AdvisorSweepReport:
    """Multi-regime crossover report across request regimes."""

    model: str
    chip: str
    parallelism: ParallelismRecommendation
    vmem_budget_mib: float
    vmem_static_mib: float
    per_token_vmem_bytes: int
    vmem_breakdown_bytes: dict[str, int]
    max_vmem_tokens: int
    ridge_dense_flop_per_byte: float
    ridge_tokens: int
    regimes: tuple[RegimeEntry, ...]
    tailoring_guide: tuple[str, ...]
    dma_buffers: int = 2
    single_layer_vmem_tokens: int = 0
    prefill_parallelism: ParallelismRecommendation | None = None
    saturation_parallelism: ParallelismRecommendation | None = None
    disagg_handoff: DisaggHandoffSpec | None = None
    chip_sources: Mapping[str, str] = field(default_factory=dict)
    model_sources: Mapping[str, str] = field(default_factory=dict)
    chip_spec: ChipSpec | None = None
    layer_group_search: LayerGroupSearchResult | None = None
    model_profile: ModelProfile | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize AdvisorSweepReport to a JSON-compatible dictionary."""
        return asdict(self)


BUCKET_SCHEDULE_SCHEMA = "advisor.bucket_schedule.v1"
BUCKET_CONSUMER_STATUS = "advisory_offline_v1"


@dataclass(frozen=True)
class BucketScheduleRule:
    """Contiguous active-token interval sharing one scope and MoE grouping mode."""

    phase: str
    layout: str
    min_active_tokens: int
    max_active_tokens: int
    effective_context: int
    recommended_scope: str
    recommended_moe_mode: str
    max_vmem_tokens: int
    num_vmem_splits: int
    step_bound_us: float
    verdict: str


@dataclass(frozen=True)
class BucketSchedulePayload:
    """Offline advisory token-bucket schedule derived from a regime sweep."""

    schema: str
    consumer_status: str
    model: str
    chip: str
    default_layout: str
    vmem_budget_mib: float
    max_vmem_tokens: int
    ridge_tokens: int
    buckets: tuple[BucketScheduleRule, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize BucketSchedulePayload to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class LayerGroupCandidate:
    """Evaluation of one layer-stack partition under fused `layer_group` pricing."""

    label: str
    layer_groups: tuple[int, ...]
    num_groups: int
    feasible: bool
    t_vmem_max: int
    n_splits: int
    step_bound_us: float
    step_achievable_us: float
    step_upper_us: float
    is_listed: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize candidate partition metrics to a JSON-ready dictionary."""
        return {
            "label": self.label,
            "layer_groups": list(self.layer_groups),
            "num_groups": self.num_groups,
            "feasible": self.feasible,
            "t_vmem_max": self.t_vmem_max,
            "n_splits": self.n_splits,
            "step_bound_us": self.step_bound_us,
            "step_achievable_us": self.step_achievable_us,
            "step_upper_us": self.step_upper_us,
            "is_listed": self.is_listed,
        }


@dataclass(frozen=True)
class LayerGroupSearchResult:
    """Comparison of candidate `layer_group` partitions against `per_op`."""

    model: str
    chip: str
    phase: Phase
    layout: str
    active_tokens: int
    per_op_step_bound_us: float
    per_op_step_achievable_us: float
    per_op_step_upper_us: float
    listed_layer_groups: tuple[int, ...]
    listed_step_bound_us: float | None
    best: LayerGroupCandidate | None
    gap_vs_per_op_us: float
    gap_vs_per_op_pct: float
    gap_vs_listed_us: float | None
    gap_vs_listed_pct: float | None
    candidates: tuple[LayerGroupCandidate, ...]
    note: str

    def to_dict(self) -> dict[str, object]:
        """Serialize layer-group search summary to a JSON-ready dictionary."""
        return {
            "model": self.model,
            "chip": self.chip,
            "phase": self.phase,
            "layout": self.layout,
            "active_tokens": self.active_tokens,
            "per_op_step_bound_us": self.per_op_step_bound_us,
            "per_op_step_achievable_us": self.per_op_step_achievable_us,
            "per_op_step_upper_us": self.per_op_step_upper_us,
            "listed_layer_groups": list(self.listed_layer_groups),
            "listed_step_bound_us": self.listed_step_bound_us,
            "best": self.best.to_dict() if self.best is not None else None,
            "gap_vs_per_op_us": self.gap_vs_per_op_us,
            "gap_vs_per_op_pct": self.gap_vs_per_op_pct,
            "gap_vs_listed_us": self.gap_vs_listed_us,
            "gap_vs_listed_pct": self.gap_vs_listed_pct,
            "candidates": [cand.to_dict() for cand in self.candidates],
            "note": self.note,
        }
