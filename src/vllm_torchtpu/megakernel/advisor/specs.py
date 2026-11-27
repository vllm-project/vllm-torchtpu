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
"""Megakernel advisor data schemas, JSON helpers, and workload specifications."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, get_args

from vllm_torchtpu.megakernel.core import errors

from .constants import (
    CALIBRATION_DIR,
    CHIP_ALIASES,
    COLLECTIVE_PATH_XLA,
    CORE_LIMITS,
    CORE_MODULE_FAMILIES,
    CORE_SUPPORTED_DTYPES,
    DTYPE_BYTES,
    EFFICIENCY_BRACKETS,
    HBM_UTILIZATION_CAP,
    MEASUREMENTS_SCHEMA,
    MIN_COLLECTIVE_GROUP_SIZE,
    MIN_VMEM_MIB,
    PUBLIC_MEASUREMENT_ALIASES,
    STRUCTURAL_LAYER_TAGS,
    SUPPORTED_ONCHIP_FAMILIES,
    TURBOQUANT_DEFAULT_PAGE_SIZE,
    VALID_COLLECTIVE_PATHS,
    VALID_DMA_BUFFERS,
    ExecutionMode,
    FusionScope,
    GainBasis,
    HostTableShardAxis,
    HyperConnectionKind,
    SparseAttnKind,
    SparseCacheReplication,
)

if TYPE_CHECKING:
    from vllm_torchtpu.megakernel.core.exec_spec import Tiles

    from .chips import ChipSpec
    from .models import ModelProfile, ParallelLayout

Phase = Literal["decode", "verify", "prefill", "mixed"]
MoEMode = Literal["masked_on_chip", "grouped_on_chip", "grouped_in_hbm", "dense_mlp"]
ProvenanceTag = Literal["DERIVED", "ASSUMED", "MEASURED"]
ObjectiveMode = Literal["step", "throughput", "both"]
DispatchRegime = Literal["auto", "zero_floor", "compiled_xla", "unfused_eager"]
SupportStatus = Literal["baseline", "within_core_limits", "outside_core_limits"]
VALID_SOURCE_SCOPES: frozenset[str] = frozenset({"", *get_args(FusionScope)})
BUCKET_SCHEDULE_SCHEMA = "advisor.bucket_schedule.v1"
BUCKET_CONSUMER_STATUS = "advisory_offline_v1"

__all__ = [
    "BUCKET_CONSUMER_STATUS",
    "BUCKET_SCHEDULE_SCHEMA",
    "CORE_MODULE_FAMILIES",
    "CORE_SUPPORTED_DTYPES",
    "DTYPE_BYTES",
    "STRUCTURAL_LAYER_TAGS",
    "SUPPORTED_ONCHIP_FAMILIES",
    "VALID_SOURCE_SCOPES",
    "AdvisorReport",
    "AdvisorSweepReport",
    "BucketScheduleEntry",
    "BucketScheduleReport",
    "CollectiveCall",
    "DisaggHandoffSpec",
    "DispatchRegime",
    "EfficiencySpec",
    "ExecutionMode",
    "FusionScope",
    "GainBasis",
    "HostTableBreakSpec",
    "HyperConnectionSpec",
    "LayerGroupCandidate",
    "LayerGroupSearchReport",
    "MeasurementsData",
    "MoEMode",
    "ObjectiveMode",
    "ParallelismComparisonRow",
    "ParallelismRecommendation",
    "Phase",
    "ProvenanceMetric",
    "ProvenanceTag",
    "RegimeEntry",
    "RingPlan",
    "ScopeEstimate",
    "SparseAttnSpec",
    "SparseCacheOwner",
    "SupportStatus",
    "WorkloadSpec",
    "align_batch_to_dp",
    "as_mapping",
    "as_scalar_number",
    "as_sequence",
    "default_double_buffer_ring_plan",
    "format_parallel_layout_label",
    "int_tuple",
    "load_measurements",
    "parse_host_table_hbm_overrides",
    "public_calibration_path",
    "scalar_bool",
    "scalar_float",
    "scalar_int",
]


PAIR_LENGTH = 2


def parse_host_table_hbm_overrides(
    raw_overrides: object, label: str = "host_table_hbm_overrides"
) -> tuple[tuple[str, bool], ...]:
    """Parse and validate `(table_name, hbm_resident)` override entries."""
    if raw_overrides is None:
        return ()
    mapping = as_mapping(raw_overrides)
    if mapping is not None:
        pairs: list[tuple[str, bool]] = []
        for key, val in mapping.items():
            if type(key) is not str or not key.strip():
                msg = f"{label} keys must be non-empty strings, got {key!r}"
                raise errors.MegakernelConfigError(msg)
            pairs.append((key.strip(), scalar_bool(val, f"{label}[{key!r}]")))
        return tuple(pairs)
    seq = as_sequence(raw_overrides)
    if seq is not None:
        pairs = []
        for idx, item in enumerate(seq):
            pair = as_sequence(item)
            if pair is None or len(pair) != PAIR_LENGTH:
                msg = f"{label}[{idx}] must be a (name, bool) pair, got {item!r}"
                raise errors.MegakernelConfigError(msg)
            raw_name, raw_flag = pair[0], pair[1]
            if type(raw_name) is not str or not raw_name.strip():
                msg = f"{label}[{idx}][0] must be a non-empty string, got {raw_name!r}"
                raise errors.MegakernelConfigError(msg)
            pairs.append(
                (raw_name.strip(), scalar_bool(raw_flag, f"{label}[{idx}][1]"))
            )
        return tuple(pairs)
    msg = f"{label} must be a mapping or sequence of (name, bool) pairs"
    raise errors.MegakernelConfigError(msg)


@dataclass(frozen=True)
class SparseCacheOwner:
    """Shared or per-layer sparse KV or indexer cache owner geometry."""

    layer: int
    record_tokens: float = 1.0
    record_elems: int = 0
    elem_bytes: float | None = None
    record_overhead_bytes: float = 0.0
    replication: SparseCacheReplication = "per_rank"
    users: tuple[int, ...] = ()
    max_scan_records: int | None = None


@dataclass(frozen=True)
class SparseAttnSpec:
    """Unified sparse-attention specification across QSA, CSA, and MSA.

    Covers Query-Selected Sparse Attention (`qsa`), Compressed Sparse
    Attention (`csa`), and Multi-Scale Sparse Attention (`msa`).
    """

    kind: SparseAttnKind
    window_tokens: int = 0
    window_bytes_per_token: float = 0.0
    window_layers: tuple[int, ...] = ()
    select_tokens: int = 0
    index_heads: int = 0
    index_kv_heads: int = 1
    index_dim: int = 0
    index_compress_ratio: int = 1
    kv_owners: tuple[SparseCacheOwner, ...] = ()
    index_owners: tuple[SparseCacheOwner, ...] = ()
    shared_kv_mqa: bool = False
    kv_owner_ring_bytes_per_req: int = 0


@dataclass(frozen=True)
class HyperConnectionSpec:
    """Multi-stream residual hyper-connection specification.

    Covers low-rank projections (`lowrank`) and Sinkhorn
    Manifold-Constrained Hyper-Connections (`mhc`).
    """

    kind: HyperConnectionKind
    num_streams: int = 4
    lowrank_dim: int = 0
    sinkhorn_iters: int = 0
    has_final_mixer: bool = False


@dataclass(frozen=True)
class HostTableBreakSpec:
    """Large host-offloaded or HBM-resident embedding lookup table specification."""

    name: str
    layers: tuple[int, ...]
    table_bytes_total: int
    h2d_bytes_per_token: float
    hbm_resident: bool = False
    shard_axis: HostTableShardAxis = "tp"
    state_bytes_per_request: int = 0
    dense_proj_params: int = 0


def as_mapping(val: object) -> Mapping[str, object] | None:
    """Return `val` when it is a `Mapping`, else `None`."""
    return val if isinstance(val, Mapping) else None


def as_sequence(val: object) -> tuple[object, ...] | None:
    """Return `val` as a tuple when it is a `list` or `tuple`, else `None`."""
    return tuple(val) if isinstance(val, (list, tuple)) else None


def as_scalar_number(val: object) -> float | None:
    """Return finite float for numeric scalars while rejecting bools and strings."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        num = float(val)
        return num if math.isfinite(num) else None
    return None


def scalar_float(val: object, field_name: str) -> float:
    """Parse a finite JSON int/float scalar, rejecting bools and non-numbers."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        msg = f"{field_name} must be numeric, got {type(val).__name__}"
        raise errors.MegakernelConfigError(msg)
    num = float(val)
    if not math.isfinite(num):
        msg = f"{field_name} must be finite, got {val!r}"
        raise errors.MegakernelConfigError(msg)
    return num


def scalar_int(val: object, field_name: str) -> int:
    """Parse an integer scalar cleanly, rejecting bools and non-integers."""
    if isinstance(val, bool) or not isinstance(val, int):
        msg = f"{field_name} must be an integer, got {type(val).__name__}"
        raise errors.MegakernelConfigError(msg)
    return val


def scalar_bool(val: object, field_name: str) -> bool:
    """Require a strict boolean scalar."""
    if type(val) is not bool:
        msg = f"{field_name} must be a boolean, got {type(val).__name__}"
        raise errors.MegakernelConfigError(msg)
    return val


def scalar_str(val: object, field_name: str) -> str:
    """Require an exact string scalar."""
    if type(val) is not str:
        msg = f"{field_name} must be a string, got {type(val).__name__}"
        raise errors.MegakernelConfigError(msg)
    return val


def optional_mapping(raw: object, field_name: str) -> Mapping[str, object]:
    """Return a Mapping or empty dict when None; reject invalid types."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return raw
    msg = f"{field_name} must be a mapping, got {type(raw).__name__}"
    raise errors.MegakernelConfigError(msg)


def int_tuple(raw: object, field_name: str) -> tuple[int, ...]:
    """Parse an integer tuple from a JSON sequence, rejecting invalid items."""
    if raw is None:
        return ()
    seq = as_sequence(raw)
    if seq is None:
        msg = f"{field_name} must be a sequence of integers"
        raise errors.MegakernelConfigError(msg)
    return tuple(scalar_int(item, f"{field_name} entries") for item in seq)


def _checked_float(
    val: object,
    field_name: str,
    *,
    lo: float = 0.0,
    hi: float | None = None,
    strict_lo: bool = False,
) -> float:
    """Parse finite float and validate optional lower/upper bounds."""
    num = scalar_float(val, field_name)
    if (num <= lo if strict_lo else num < lo) or (hi is not None and num > hi):
        op = "(" if strict_lo else "["
        bound = (
            f"{op}{lo:g}, {hi:g}]"
            if hi is not None
            else f"{'>' if strict_lo else '>='} {lo:g}"
        )
        msg = f"{field_name} must be in {bound}, got {num}"
        raise errors.MegakernelConfigError(msg)
    return num


def _opt_float(
    val: object, field_name: str, *, strict_lo: bool = False
) -> float | None:
    """Parse an optional non-negative or strictly positive finite float."""
    return None if val is None else _checked_float(val, field_name, strict_lo=strict_lo)


def _checked_int(val: object, field_name: str, *, lo: int = 0) -> int:
    """Parse integer scalar and enforce `>= lo`."""
    num = scalar_int(val, field_name)
    if num < lo:
        msg = f"{field_name} must be >= {lo}, got {num}"
        raise errors.MegakernelConfigError(msg)
    return num


def public_calibration_path(
    chip_name: str = "tpu7x", *, calibration_dir: Path | None = None
) -> Path | None:
    """Return path to bundled architectural priors JSON for `chip_name` if present."""
    base_dir = CALIBRATION_DIR if calibration_dir is None else calibration_dir
    raw = chip_name.strip().lower().removeprefix("tpu_").removesuffix("_megacore")
    candidate = base_dir / f"{CHIP_ALIASES.get(raw, raw)}_public.json"
    return candidate if candidate.is_file() else None


class ProvenanceMetric(TypedDict):
    """Single output metric value paired with its provenance tag, source, and band."""

    value: float | int | bool | str
    unit: str
    tag: ProvenanceTag
    source: str
    band: list[float]


@dataclass(frozen=True)
class RingPlan:
    """On-chip VMEM ring-buffer slot counts and bank widths for weight streaming."""

    dense_slots: int = 2
    expert_slots: int = 2
    dense_slot_bytes: int = CORE_LIMITS.dense_slot_bytes
    expert_slot_bytes: int = CORE_LIMITS.expert_slot_bytes
    compiler_spill_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        """Validate positive VMEM ring slot counts and byte widths."""
        for name, lo in (
            ("dense_slots", 1),
            ("expert_slots", 0),
            ("dense_slot_bytes", 1),
            ("expert_slot_bytes", 0),
            ("compiler_spill_reserve_bytes", 0),
        ):
            _checked_int(getattr(self, name), f"RingPlan.{name}", lo=lo)


def default_double_buffer_ring_plan(dma_buffers: int = 2) -> RingPlan:
    """Return the default n-slot weight-streaming RingPlan."""
    return RingPlan(dense_slots=dma_buffers, expert_slots=dma_buffers)


@dataclass(frozen=True)
class CollectiveCall:
    """One measured collective: where it ran, which op, across how many ranks."""

    path: str  # "xla" (an XLA op) or "in_kernel" (inside a fused call)
    op: str
    group_size: int
    latency_us: float
    n: int
    mean_latency_us: float | None = None
    bytes: int | None = None


@dataclass(frozen=True)
class MeasurementsData:
    """Calibrated runtime measurements loaded from an `advisor.measurements.v1` file."""

    source_desc: str
    source_kind: str = "trace"
    source_image: str = ""
    source_scope: str = ""
    launch_gap_p50_us: float | None = None
    launch_gap_p90_us: float | None = None
    launch_gap_n: int = 0
    collective_calls: tuple[CollectiveCall, ...] = ()
    step_coverage: float | None = None
    step_time_us: float | None = None
    per_op_step_us: float | None = None
    host_head_p50_us: float | None = None
    host_head_p90_us: float | None = None
    host_head_n: int = 0
    dcn_bw_gbps: float | None = None
    ring_plan: RingPlan | None = None


def _parse_measurements_source(
    payload: Mapping[str, object], file_label: str
) -> tuple[str, str, str, str]:
    """Parse and validate the `source` metadata block in measurements JSON."""
    source_map = optional_mapping(payload.get("source"), "source")
    kind = scalar_str(source_map.get("kind", "trace"), "source.kind")
    src_file = scalar_str(source_map.get("file", file_label), "source.file")
    image = scalar_str(source_map.get("image", ""), "source.image").strip()
    scope = scalar_str(source_map.get("scope", ""), "source.scope").strip()
    if scope not in VALID_SOURCE_SCOPES:
        msg = (
            f"source.scope must be one of {sorted(VALID_SOURCE_SCOPES)}, got {scope!r}"
        )
        raise errors.MegakernelConfigError(msg)
    return f"{kind}:{src_file}", kind, image, scope


def _parse_one_collective_call(idx: int, entry: Mapping[str, object]) -> CollectiveCall:
    """Validate and normalize one collective call mapping from JSON."""
    prefix = f"collectives[{idx}]"
    if "latency_us" not in entry:
        msg = f"{prefix} missing required 'latency_us'"
        raise errors.MegakernelConfigError(msg)
    latency_us = _checked_float(entry["latency_us"], f"{prefix}.latency_us")
    group_size = _checked_int(
        entry.get("group_size", MIN_COLLECTIVE_GROUP_SIZE),
        f"{prefix}.group_size",
        lo=MIN_COLLECTIVE_GROUP_SIZE,
    )
    sample_count = _checked_int(entry.get("n", 1), f"{prefix}.n", lo=0)
    raw_bytes = entry.get("bytes")
    msg_bytes = (
        None if raw_bytes is None else _checked_int(raw_bytes, f"{prefix}.bytes", lo=0)
    )
    op_raw = entry.get("op")
    op_norm = (
        ""
        if op_raw is None
        else scalar_str(op_raw, f"{prefix}.op").strip().lower().replace("_", "-")
    )
    path_raw = entry.get("path")
    path_val = (
        COLLECTIVE_PATH_XLA
        if path_raw is None
        else scalar_str(path_raw, f"{prefix}.path")
    )
    if path_val not in VALID_COLLECTIVE_PATHS:
        valid = sorted(VALID_COLLECTIVE_PATHS)
        msg = f"{prefix}.path must be in {valid}, got {path_val!r}"
        raise errors.MegakernelConfigError(msg)
    mean_us = _opt_float(entry.get("mean_latency_us"), f"{prefix}.mean_latency_us")
    return CollectiveCall(
        path=path_val,
        op=op_norm,
        group_size=group_size,
        latency_us=latency_us,
        n=sample_count,
        mean_latency_us=mean_us,
        bytes=msg_bytes,
    )


def _parse_collective_calls(raw_collectives: object) -> tuple[CollectiveCall, ...]:
    """Parse and normalize measured collective call entries from JSON."""
    if raw_collectives is None:
        return ()
    if type(raw_collectives) is not list:
        msg = f"collectives must be a list, got {type(raw_collectives).__name__}"
        raise errors.MegakernelConfigError(msg)
    calls: list[CollectiveCall] = []
    for idx, item in enumerate(raw_collectives):
        entry = as_mapping(item)
        if entry is None:
            msg = f"collectives[{idx}] must be a mapping"
            raise errors.MegakernelConfigError(msg)
        calls.append(_parse_one_collective_call(idx, entry))
    return tuple(calls)


def _parse_ring_plan(ring_plan_raw: object) -> RingPlan | None:
    """Parse an optional RingPlan mapping from measurements JSON."""
    if ring_plan_raw is None:
        return None
    ring_map = optional_mapping(ring_plan_raw, "ring_plan")
    defaults = (
        ("dense_slots", 2),
        ("expert_slots", 2),
        ("dense_slot_bytes", CORE_LIMITS.dense_slot_bytes),
        ("expert_slot_bytes", CORE_LIMITS.expert_slot_bytes),
        ("compiler_spill_reserve_bytes", 0),
    )
    vals = {k: scalar_int(ring_map.get(k, d), f"ring_plan.{k}") for k, d in defaults}
    return RingPlan(**vals)


def _load_measurements_payload(
    target: object, chip_name: str
) -> tuple[Mapping[str, object], str]:
    """Resolve and validate the top-level measurements JSON mapping."""
    if isinstance(target, Mapping):
        return target, "inline_mapping"
    if not isinstance(target, (str, Path)):
        msg = (
            "Measurements target must be a path or mapping, "
            f"got {type(target).__name__}"
        )
        raise errors.MegakernelConfigError(msg)
    target_str = str(target).strip()
    if not target_str:
        msg = "Measurements target path must not be empty"
        raise errors.MegakernelConfigError(msg)
    if target_str in PUBLIC_MEASUREMENT_ALIASES:
        path = public_calibration_path(chip_name)
        if path is None or not path.is_file():
            msg = f"No bundled architectural priors found for chip {chip_name!r}"
            raise errors.MegakernelConfigError(msg)
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
    if not isinstance(parsed, Mapping):
        msg = f"Measurements file {path} must contain a JSON object"
        raise errors.MegakernelConfigError(msg)
    return parsed, path.name


def _parse_nonneg_quantiles(
    payload: Mapping[str, object], field_name: str
) -> tuple[float | None, float | None, int]:
    """Parse optional `{p50, p90, n}` quantile statistics in microseconds."""
    stats_map = optional_mapping(payload.get(field_name), field_name)
    if not stats_map:
        return None, None, 0
    if "p50" not in stats_map:
        msg = f"{field_name} requires 'p50' when non-empty"
        raise errors.MegakernelConfigError(msg)
    p50_us = _checked_float(stats_map["p50"], f"{field_name}.p50")
    p90_us = _checked_float(stats_map.get("p90", p50_us), f"{field_name}.p90")
    if p90_us < p50_us:
        msg = f"{field_name}.p90 ({p90_us}) must be >= p50 ({p50_us})"
        raise errors.MegakernelConfigError(msg)
    return p50_us, p90_us, _checked_int(stats_map.get("n", 0), f"{field_name}.n", lo=0)


def _parse_step_time(
    step_time_raw: object,
    field_name: str = "step_time_us",
    *,
    strictly_positive: bool = False,
) -> float | None:
    """Parse optional step latency scalar or `{p50: ...}` mapping."""
    if step_time_raw is None:
        return None
    if isinstance(step_time_raw, Mapping):
        if not step_time_raw:
            return None
        if "p50" not in step_time_raw:
            msg = f"{field_name} mapping requires 'p50'"
            raise errors.MegakernelConfigError(msg)
        return _checked_float(
            step_time_raw["p50"], f"{field_name}.p50", strict_lo=strictly_positive
        )
    return _checked_float(step_time_raw, field_name, strict_lo=strictly_positive)


def load_measurements(target: object, *, chip_name: str = "tpu7x") -> MeasurementsData:
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
    src_desc, src_kind, src_image, src_scope = _parse_measurements_source(
        payload, file_label
    )
    lg_p50, lg_p90, lg_n = _parse_nonneg_quantiles(payload, "launch_gap_us")
    head_p50, head_p90, head_n = _parse_nonneg_quantiles(payload, "host_head_us")
    coverage_map = optional_mapping(payload.get("step_coverage"), "step_coverage")
    step_cov = (
        _checked_float(
            coverage_map["fused_eligible_fraction"],
            "step_coverage.fused_eligible_fraction",
            hi=1.0,
            strict_lo=True,
        )
        if "fused_eligible_fraction" in coverage_map
        else None
    )
    return MeasurementsData(
        source_desc=src_desc,
        source_kind=src_kind,
        source_image=src_image,
        source_scope=src_scope,
        launch_gap_p50_us=lg_p50,
        launch_gap_p90_us=lg_p90,
        launch_gap_n=lg_n,
        collective_calls=_parse_collective_calls(payload.get("collectives")),
        step_coverage=step_cov,
        step_time_us=_parse_step_time(
            payload.get("step_time_us"), "step_time_us", strictly_positive=True
        ),
        per_op_step_us=_parse_step_time(
            payload.get("per_op_step_us"), "per_op_step_us", strictly_positive=True
        ),
        host_head_p50_us=head_p50,
        host_head_p90_us=head_p90,
        host_head_n=head_n,
        dcn_bw_gbps=_opt_float(
            payload.get("dcn_bw_gbps"), "dcn_bw_gbps", strict_lo=True
        ),
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
        for name in ("eta_hbm", "eta_dense", "eta_moe_grouped", "eta_ici"):
            _checked_float(
                getattr(self, name), f"EfficiencySpec.{name}", hi=1.0, strict_lo=True
            )


def align_batch_to_dp(batch: int, dp: int) -> int:
    """Round `batch` up to the smallest positive multiple of `dp`."""
    dp_ranks = max(1, dp)
    return dp_ranks * max(1, (batch + dp_ranks - 1) // dp_ranks)


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
    hbm_utilization_cap: float = HBM_UTILIZATION_CAP
    dispatch_regime: DispatchRegime = "auto"
    efficiency: EfficiencySpec = EfficiencySpec()
    # Most rows a single decode or verify call executes before tiling across calls.
    # 0 uses CORE_LIMITS.max_batch_rows as the hardware ceiling.
    max_rows_per_call: int = 0
    host_head_per_op_us: float = 0.0
    host_head_fused_us: float = 0.0
    host_head_explicit: bool = False
    per_op_step_us: float | None = None
    step_coverage: float | None = None
    batch_explicit: bool = False
    # True when the caller explicitly fixed moe_mode rather than leaving it unpinned.
    moe_mode_explicit: bool = False
    latent_proj_tp: int = 0
    # Optional explicit step execution override (`None` inherits `layout_execution`).
    execution: ExecutionMode | None = None
    # Optional explicit per-table HBM overrides (`None` inherits layout overrides;
    # `()` explicitly clears layout overrides).
    host_table_hbm_overrides: tuple[tuple[str, bool], ...] | None = None
    # Layout-level execution default set by `with_layout`.
    layout_execution: ExecutionMode = "full"
    # Layout-level host-table HBM defaults set by `with_layout`.
    layout_host_table_hbm_overrides: tuple[tuple[str, bool], ...] = ()
    kv_page_size: int = TURBOQUANT_DEFAULT_PAGE_SIZE

    @property
    def effective_execution(self) -> ExecutionMode:
        """Return explicit user execution override if set, else layout default."""
        return self.execution if self.execution is not None else self.layout_execution

    @property
    def effective_host_table_hbm_overrides(self) -> tuple[tuple[str, bool], ...]:
        """Return effective host-table HBM overrides merged over layout defaults."""
        if self.host_table_hbm_overrides is None:
            return self.layout_host_table_hbm_overrides
        if not self.host_table_hbm_overrides:
            return ()
        merged = dict(self.layout_host_table_hbm_overrides)
        merged.update(self.host_table_hbm_overrides)
        return tuple(merged.items())

    @staticmethod
    def _validate_table_overrides(overrides: object, field_name: str) -> None:
        """Validate `tuple[tuple[str, bool], ...]` host-table override pairs."""
        if type(overrides) is not tuple:
            msg = f"{field_name} must be a tuple"
            raise errors.MegakernelConfigError(msg)
        for idx, entry in enumerate(overrides):
            if type(entry) is not tuple or len(entry) != PAIR_LENGTH:
                msg = f"{field_name}[{idx}] must be a (str, bool) tuple, got {entry!r}"
                raise errors.MegakernelConfigError(msg)
            table_name, is_hbm = entry
            if type(table_name) is not str or not table_name.strip():
                msg = (
                    f"{field_name}[{idx}][0] must be a non-empty string, "
                    f"got {table_name!r}"
                )
                raise errors.MegakernelConfigError(msg)
            scalar_bool(is_hbm, f"{field_name}[{idx}][1]")

    def _validate_literals_and_bools(self) -> None:
        """Enforce exact string literals and strict boolean flags."""
        literal_checks: tuple[tuple[str, tuple[object, ...]], ...] = (
            ("phase", get_args(Phase)),
            ("moe_mode", get_args(MoEMode)),
            ("objective", get_args(ObjectiveMode)),
            ("dispatch_regime", get_args(DispatchRegime)),
            ("layout_execution", get_args(ExecutionMode)),
        )
        for name, allowed in literal_checks:
            val = getattr(self, name)
            if type(val) is not str or val not in allowed:
                msg = f"WorkloadSpec.{name} must be one of {list(allowed)}, got {val!r}"
                raise errors.MegakernelConfigError(msg)
        allowed_exec = get_args(ExecutionMode)
        if self.execution is not None and (
            type(self.execution) is not str or self.execution not in allowed_exec
        ):
            msg = (
                f"WorkloadSpec.execution must be one of {list(allowed_exec)}, "
                f"got {self.execution!r}"
            )
            raise errors.MegakernelConfigError(msg)
        for name in (
            "vmem_alias",
            "vmem_fit_headroom",
            "host_head_explicit",
            "batch_explicit",
            "moe_mode_explicit",
        ):
            scalar_bool(getattr(self, name), f"WorkloadSpec.{name}")
        if self.host_table_hbm_overrides is not None:
            self._validate_table_overrides(
                self.host_table_hbm_overrides, "WorkloadSpec.host_table_hbm_overrides"
            )
        self._validate_table_overrides(
            self.layout_host_table_hbm_overrides,
            "WorkloadSpec.layout_host_table_hbm_overrides",
        )

    def __post_init__(self) -> None:
        """Enforce single-field literal, bool, integer, and finite float invariants."""
        self._validate_literals_and_bools()
        pos_ints = (
            "batch",
            "tp",
            "pp",
            "dp",
            "ep",
            "kv_split",
            "dcp",
            "pcp",
            "kv_page_size",
        )
        nonneg_ints = (
            "isl",
            "osl",
            "context",
            "spec_tokens",
            "chunk_size",
            "microbatches",
            "segment_layers",
            "max_rows_per_call",
            "latent_proj_tp",
        )
        for lo, names in ((1, pos_ints), (0, nonneg_ints)):
            for name in names:
                _checked_int(getattr(self, name), f"WorkloadSpec.{name}", lo=lo)
        for idx, count in enumerate(self.layer_groups):
            _checked_int(count, f"WorkloadSpec.layer_groups[{idx}]", lo=1)
        if (
            type(self.dma_buffers) is not int
            or self.dma_buffers not in VALID_DMA_BUFFERS
        ):
            msg = (
                f"WorkloadSpec.dma_buffers must be in {VALID_DMA_BUFFERS}, "
                f"got {self.dma_buffers!r}"
            )
            raise errors.MegakernelConfigError(msg)
        _checked_float(self.prefix_hit_rate, "WorkloadSpec.prefix_hit_rate")
        if self.prefix_hit_rate >= 1.0:
            msg = (
                "WorkloadSpec.prefix_hit_rate must be in [0, 1), "
                f"got {self.prefix_hit_rate}"
            )
            raise errors.MegakernelConfigError(msg)
        float_checks: tuple[tuple[str, float, float | None, bool], ...] = (
            ("hbm_utilization_cap", 0.0, 1.0, True),
            ("vmem_budget_mib", float(MIN_VMEM_MIB), None, False),
            ("op_overhead_us", 0.0, None, False),
            ("host_head_per_op_us", 0.0, None, False),
            ("host_head_fused_us", 0.0, None, False),
            ("per_op_step_us", 0.0, None, True),
            ("step_coverage", 0.0, 1.0, True),
        )
        for name, lo, hi, strict_lo in float_checks:
            val = getattr(self, name)
            if val is not None:
                _checked_float(
                    val, f"WorkloadSpec.{name}", lo=lo, hi=hi, strict_lo=strict_lo
                )

    def with_layout(
        self,
        layout: ParallelLayout | ParallelismRecommendation,
        *,
        align_batch: bool = False,
    ) -> WorkloadSpec:
        """Return a copy with layout parallelism and execution defaults applied."""
        return replace(
            self,
            batch=(
                align_batch_to_dp(self.batch, layout.dp) if align_batch else self.batch
            ),
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
            layer_groups=layout.layer_groups or self.layer_groups,
            layout_execution=layout.execution,
            layout_host_table_hbm_overrides=layout.host_table_hbm_overrides,
        )


@dataclass(frozen=True)
class ScopeEstimate:
    """Roofline and VMEM capacity estimate for one fusion scope."""

    scope: FusionScope
    feasible: bool
    reason: str
    ops_per_step: int
    vmem_static_mib: float
    max_vmem_tokens: int
    num_vmem_splits: int
    step_weight_gb: float
    critical_weight_gb: float | None
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
    launch_us: float = 0.0
    step_achievable_upper_us: float = 0.0
    support: SupportStatus = "outside_core_limits"
    blockers: tuple[str, ...] = ()
    step_weight_strict_gb: float = 0.0
    step_floor_strict_us: float = 0.0
    # Dense compute plus the routed-expert stage, max(routed HBM, MoE compute).
    # The core is max(hbm_us, chain_us + comm_us) when collectives overlap
    # inside one fused call (num_vmem_splits == 1), else
    # max(hbm_us, chain_us) + comm_us.
    chain_us: float = 0.0
    expert_mode: str = "none"
    host_head_us: float = 0.0
    break_even_host_head_delta_us: float = 0.0
    ranking_metric: str = "step_bound_us"
    binding_unit: str = "hbm"


class ParallelismComparisonRow(TypedDict):
    """Single layout comparison entry returned by `compare_parallelism_candidates`."""

    layout: str
    chips: int
    step_floor_us: float
    throughput_tok_s_per_chip: float
    recommended_scope: FusionScope
    fastest_scope: FusionScope
    hbm_fits: bool
    valid: bool


@dataclass(frozen=True)
class AdvisorReport:
    """First-principles physics fusion-scope verdict for a single workload.

    `hbm_fits` checks physical HBM capacity (`<= 1.00 * hbm_capacity_bytes`)
    to report hard physical OOM, whereas `hbm_pool_overflow` flags footprints
    exceeding `workload.hbm_utilization_cap` (defaulting to
    `HBM_UTILIZATION_CAP = 0.92` for non-KV runtime headroom).
    """

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
    gain_basis: GainBasis = "fused_scope_device_estimate"
    coverage: str = (
        "decoder layer stack; excludes embedding, LM head, sampling, "
        "draft heads and modality towers"
    )
    hbm_pool_overflow: bool = False
    hbm_utilization_cap: float = HBM_UTILIZATION_CAP
    ranking_metric: str = "step_bound_us"
    binding_unit: str = "hbm"
    core_support: SupportStatus = "baseline"
    core_blockers: tuple[str, ...] = ()
    core_ready: bool = False
    core_ready_scope: FusionScope = "per_op"

    def to_dict(self) -> dict[str, object]:
        """Serialize `AdvisorReport` to a JSON-compatible dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return asdict(self)


@dataclass(frozen=True)
class LayerGroupCandidate:
    """Evaluation of one layer-stack partition under fused `layer_group` pricing."""

    label: str
    layer_groups: tuple[int, ...]
    num_groups: int
    feasible: bool
    max_vmem_tokens: int
    num_vmem_splits: int
    step_bound_us: float
    step_achievable_us: float
    step_upper_us: float
    is_listed: bool = False


@dataclass(frozen=True)
class LayerGroupSearchReport:
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
        return asdict(self)


def format_parallel_layout_label(
    layout: WorkloadSpec | ParallelismRecommendation | ParallelLayout,
) -> str:
    """Return layout axes as `TP8 x EP8 x DCP4`, omitting degree-1 axes."""
    axes = (
        ("TP", layout.tp),
        ("DP", layout.dp),
        ("EP", layout.ep),
        ("PP", layout.pp),
        ("PCP", layout.pcp),
        ("DCP", layout.dcp),
    )
    return " x ".join(f"{name}{deg}" for name, deg in axes if deg > 1) or "TP1"


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
    step_weight_gb: float
    auto_selected: bool = True
    dcp: int = 1
    pcp: int = 1
    source: str = ""
    layer_groups: tuple[int, ...] = ()
    execution: ExecutionMode = "full"
    host_table_hbm_overrides: tuple[tuple[str, bool], ...] = ()

    def label(self) -> str:
        """Return the layout as `TP8 x EP8 x DCP4`, omitting degree-1 axes."""
        return format_parallel_layout_label(self)


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
    hbm_fits: bool = True
    hbm_pool_overflow: bool = False
    flags: tuple[str, ...] = ()
    metrics: dict[str, ProvenanceMetric] = field(default_factory=dict)
    unresolved_assumptions: tuple[dict[str, object], ...] = ()
    # Serialized ScopeEstimate dicts across all evaluated fusion scopes.
    scopes: tuple[dict[str, object], ...] = ()
    workload: dict[str, object] = field(default_factory=dict)
    fastest_scope: FusionScope = "per_op"
    fastest_verdict: str = "separate_likely"
    # Flat JSON alias mirroring `core_support`.
    evidence: SupportStatus = "baseline"
    step_achievable_us: float = 0.0
    step_achievable_upper_us: float = 0.0
    step_floor_strict_us: float = 0.0
    # Flat JSON alias mirroring `core_blockers`.
    blockers: tuple[str, ...] = ()
    gain_basis: GainBasis = "fused_scope_device_estimate"
    ranking_metric: str = "step_bound_us"
    binding_unit: str = "hbm"
    core_support: SupportStatus = "baseline"
    core_blockers: tuple[str, ...] = ()
    core_ready: bool = False
    core_ready_scope: FusionScope = "per_op"

    def to_dict(self) -> dict[str, object]:
        """Serialize `RegimeEntry` to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class AdvisorSweepReport:
    """Multi-regime physics crossover report across canonical request regimes."""

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
    vmem_reference_scope: FusionScope = "whole_step"
    prefill_parallelism: ParallelismRecommendation | None = None
    saturation_parallelism: ParallelismRecommendation | None = None
    disagg_handoff: DisaggHandoffSpec | None = None
    chip_sources: Mapping[str, str] = field(default_factory=dict)
    model_sources: Mapping[str, str] = field(default_factory=dict)
    chip_spec: ChipSpec | None = None
    layer_group_search: LayerGroupSearchReport | None = None
    model_profile: ModelProfile | None = None
    submodule_labels: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize `AdvisorSweepReport` to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["regimes"] = [entry.to_dict() for entry in self.regimes]
        return payload


@dataclass(frozen=True)
class BucketScheduleEntry:
    """Contiguous active-token interval sharing scope, MoE mode, and readiness.

    When multiple sweep rows share `(recommended_scope, recommended_moe_mode,
    verdict, core_support, core_ready_scope, core_blockers)` across context
    lengths, `build_bucket_schedule` coalesces them into one interval whose
    `effective_context` records the maximum context across the merged rows.
    """

    phase: Phase
    layout: str
    min_active_tokens: int
    max_active_tokens: int
    effective_context: int
    recommended_scope: FusionScope
    recommended_moe_mode: str
    max_vmem_tokens: int
    num_vmem_splits: int
    step_bound_us: float
    verdict: str
    core_support: SupportStatus = "baseline"
    core_ready_scope: FusionScope = "per_op"
    core_blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize `BucketScheduleEntry` to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["core_blockers"] = list(self.core_blockers)
        return payload


@dataclass(frozen=True)
class BucketScheduleReport:
    """Offline advisory token-bucket schedule derived from a regime sweep."""

    schema: str
    consumer_status: str
    model: str
    chip: str
    default_layout: str
    vmem_budget_mib: float
    max_vmem_tokens: int
    ridge_tokens: int
    buckets: tuple[BucketScheduleEntry, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize `BucketScheduleReport` to a JSON-compatible dictionary."""
        return {
            **asdict(self),
            "buckets": [entry.to_dict() for entry in self.buckets],
        }
