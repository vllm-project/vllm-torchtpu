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
"""Trace calibration engine for TPU megakernel advisor priors."""

from __future__ import annotations

import gzip
import json
from bisect import bisect_left
from dataclasses import dataclass, field
from decimal import Decimal
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast, get_args

from vllm_torchtpu.megakernel.core import errors

from .chips import resolve_chip_spec
from .constants import (
    COLLECTIVE_IGNORE_SUFFIXES,
    COLLECTIVE_PREFIXES,
    CONSERVED_GAP_DECIMALS,
    CONSERVED_GAP_UNIT,
    DEFAULT_CHIP,
    DEFAULT_PHASE,
    FUSED_ELIGIBLE_PHASES,
    MAX_INTRA_STEP_GAP_US,
    MEASUREMENTS_SCHEMA,
    MIN_COLLECTIVE_GROUP_SIZE,
    MIN_EVENTS_FOR_WARMUP_DROP,
    MIN_STEP_OPS_FOR_EXPLICIT_NORM,
    MIN_STEPS_FOR_WARMUP_DROP,
    MIN_WARMUP_DROP_STEPS,
    NS_PER_US,
    PS_PER_US,
    QUANTILE_DECIMALS,
    STEP_COVERAGE_BASIS,
    SUBLAYER_BOUNDARIES_PER_LAYER,
    WARMUP_OUTLIER_RATIO,
    XLA_MODULE_MARKERS,
    XPLANE_DEVICE_PLANE_PREFIX,
    XPLANE_SPARSE_FILTER,
)
from .models import resolve_model_profile
from .specs import Phase, as_mapping, as_scalar_number, as_sequence

# Heuristic 12-layer boundary floor separating full traces from small test fixtures.
CONSERVED_TRACE_MIN_OPS = SUBLAYER_BOUNDARIES_PER_LAYER * 12
XPLANE_COMPUTE_LINES: tuple[str, ...] = ("xla ops", "async xla ops")
PERFETTO_COMPUTE_CATEGORIES: frozenset[str] = frozenset(
    {"op", "xla_op", "xla_ops", "kernel", "compute"}
)
XPLANE_ANNOTATION_LINE_MARKERS: tuple[str, ...] = (
    "name scope",
    "source code",
    "traceme",
    "framework",
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Protocol

    class _XStatProto(Protocol):
        """Structural protocol for an XStat protobuf message."""

        metadata_id: int
        int64_value: int
        uint64_value: int
        double_value: float
        str_value: str
        bytes_value: bytes
        ref_value: int
        WhichOneof: Callable[[str], str | None]

    class _XStatMetaProto(Protocol):
        """Structural protocol for an XStatMetadata protobuf message."""

        name: str

    class _XEventMetaProto(Protocol):
        """Structural protocol for an XEventMetadata protobuf message."""

        name: str
        display_name: str
        stats: Sequence[_XStatProto]

    class _XEventProto(Protocol):
        """Structural protocol for an XEvent protobuf message."""

        metadata_id: int
        offset_ps: int
        duration_ps: int
        stats: Sequence[_XStatProto]
        WhichOneof: Callable[[str], str | None]

    class _XLineProto(Protocol):
        """Structural protocol for an XLine protobuf message."""

        id: int
        name: str
        display_name: str
        timestamp_ns: int
        events: Sequence[_XEventProto]

    class _XPlaneProto(Protocol):
        """Structural protocol for an XPlane protobuf message."""

        id: int
        name: str
        lines: Sequence[_XLineProto]
        event_metadata: Mapping[int, _XEventMetaProto]
        stat_metadata: Mapping[int, _XStatMetaProto]

    class _XSpaceProto(Protocol):
        """Structural protocol for an XSpace protobuf message."""

        planes: Sequence[_XPlaneProto]
        ParseFromString: Callable[[bytes], int]

    class _XPlaneModuleProto(Protocol):
        """Structural protocol for the generated `xplane_pb2` module."""

        XSpace: Callable[[], _XSpaceProto]


try:
    from google.protobuf.message import DecodeError

    from vllm_torchtpu.tools import merge_xplane, xplane_pb2
except ImportError:
    DecodeError = ValueError
    merge_xplane = None
    xplane_pb2 = None


class XPlaneToolUnavailableError(errors.MegakernelConfigError):
    """Raised when xplane tools (`merge_xplane` or `xplane_pb2`) are unavailable."""


@dataclass(frozen=True)
class CalibrateOptions:
    """Typed calibration provenance and layer-normalization options."""

    chip: str = DEFAULT_CHIP
    model: str | None = None
    phase: str = DEFAULT_PHASE
    batch: int | None = None
    tp: int | None = None
    num_layers: int | None = None

    def __post_init__(self) -> None:
        """Validate phase, batch, TP, and layer-count bounds upfront."""
        if self.phase not in get_args(Phase):
            msg = f"phase must be one of {sorted(get_args(Phase))}, got {self.phase!r}"
            raise errors.MegakernelConfigError(msg)
        for name, val in (("batch", self.batch), ("num_layers", self.num_layers)):
            if val is not None and (type(val) is bool or val < 1):
                msg = f"{name} must be >= 1, got {val}"
                raise errors.MegakernelConfigError(msg)
        if self.tp is not None and (
            type(self.tp) is bool or self.tp < 1 or (self.tp & (self.tp - 1)) != 0
        ):
            msg = f"tp must be a power of 2 >= 1, got {self.tp}"
            raise errors.MegakernelConfigError(msg)


class QuantileSummary(TypedDict, total=False):
    """Summary statistics (`p50`, `p90`, sample count `n`) for a latency series."""

    p50: float
    p90: float
    n: int
    step_idle_gap_us: float
    num_layers_norm: int
    unit: str


class CollectiveSummary(TypedDict):
    """Calibrated latency summary for one `(op, group_size, bytes)` bucket."""

    op: str
    group_size: int
    bytes: int
    latency_us: float
    mean_latency_us: float
    n: int


class StepCoverageSummary(TypedDict):
    """Calibrated fraction of tokens eligible for fused megakernel execution."""

    fused_eligible_fraction: float
    basis: str
    n: int


class CalibrationSourceMeta(TypedDict, total=False):
    """Provenance metadata for the input trace used during calibration."""

    kind: str
    file: str
    chip: str
    phase: str
    model: str
    batch: int
    tp: int


class CalibrationPayload(TypedDict, total=False):
    """Schema `advisor.measurements.v1` JSON payload returned by `calibrate_trace`."""

    schema: str
    source: CalibrationSourceMeta
    launch_gap_us: QuantileSummary
    collectives: list[CollectiveSummary]
    step_coverage: StepCoverageSummary
    step_time_us: QuantileSummary
    host_head_us: QuantileSummary


@dataclass(frozen=True, slots=True)
class _StepWindow:
    """Normalized execution step interval with optional token and host-head data."""

    start_us: float
    end_us: float
    duration_us: float
    phase: str | None = None
    active_tokens: int | None = None
    eligible_tokens: int | None = None
    host_head_us: float | None = None


@dataclass(frozen=True, slots=True)
class _ComputeEvent:
    """Normalized compute or collective trace event scoped to a rank stream."""

    rank_key: str
    start_us: float
    end_us: float
    duration_us: float
    collective_op: str | None = None
    group_size: int | None = None
    byte_size: int = 0


@dataclass(frozen=True, slots=True)
class _ModuleWindow:
    """Raw `XLA Modules` program wrapper span on a device rank."""

    program_name: str
    start_us: float
    end_us: float


@dataclass(frozen=True, slots=True)
class _ParsedTrace:
    """Intermediate normalized representation of an XPlane or Perfetto trace."""

    kind: str
    file_label: str
    steps_by_rank: dict[str, list[_StepWindow]] = field(default_factory=dict)
    events_by_stream: dict[str, list[_ComputeEvent]] = field(default_factory=dict)
    modules_by_rank: dict[str, list[_ModuleWindow]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PlaneIngestContext:
    """Per-plane metadata maps and origin context for XPlane line ingestion."""

    rank_key: str
    origin_ns: int
    has_xla_ops_line: bool
    event_metadata: Mapping[int, _XEventMetaProto]
    stat_metadata: Mapping[int, _XStatMetaProto]
    meta_stats: Mapping[int, dict[str, object]]


def _raw_percentile(sorted_vals: Sequence[float], quantile: float) -> float:
    """Compute unrounded linear-interpolated `quantile` over sorted values."""
    if not sorted_vals:
        return 0.0
    pos = quantile * (len(sorted_vals) - 1)
    lower_idx = int(pos)
    upper_idx = min(len(sorted_vals) - 1, lower_idx + 1)
    frac = pos - lower_idx
    return sorted_vals[lower_idx] * (1.0 - frac) + sorted_vals[upper_idx] * frac


def _percentile(
    sorted_vals: Sequence[float],
    quantile: float,
    decimals: int = QUANTILE_DECIMALS,
) -> float:
    """Compute linear-interpolated `quantile` rounded to `decimals` digits."""
    return round(_raw_percentile(sorted_vals, quantile), decimals)


def _positive_percentile(
    sorted_vals: Sequence[float],
    quantile: float,
    decimals: int = QUANTILE_DECIMALS,
) -> float:
    """Round `quantile`, falling back to unrounded value if `< 0.5 ulp` rounds to 0."""
    raw_value = _raw_percentile(sorted_vals, quantile)
    rounded = round(raw_value, decimals)
    return rounded if rounded > 0.0 else raw_value


def _classify_collective(name: str) -> str | None:
    """Return canonical collective op name if `name` is a collective event."""
    normalized = name.lower().replace("_", "-")
    if any(suffix in normalized for suffix in COLLECTIVE_IGNORE_SUFFIXES):
        return None
    for prefix, canonical in COLLECTIVE_PREFIXES:
        if prefix in normalized:
            return canonical
    return None


def _is_module_wrapper(name: str, category: str, line_name: str) -> bool:
    """Return True when an event or line wraps full-step XLA modules."""
    cat_lower, line_lower = category.lower(), line_name.lower()
    return any(
        marker in line_lower or marker in cat_lower for marker in XLA_MODULE_MARKERS
    ) or name.lower().startswith("xla_module")


def _is_step_event(
    name: str, category: str, line_name: str, args: Mapping[str, object]
) -> bool:
    """Return True when an event marks a model execution step window."""
    name_lower = name.lower()
    return (
        "step" in category.lower()
        or "step" in line_name.lower()
        or name_lower.startswith("step")
        or name_lower == "model_step"
        or "step" in args
        or "step_num" in args
    )


def _int_or_none(value: object) -> int | None:
    """Convert a finite numeric or ASCII digit-string scalar (excluding bool) to int."""
    if type(value) is int:
        return value
    if (numeric := as_scalar_number(value)) is not None:
        return int(numeric)
    if type(value) is str:
        stripped = value.strip()
        digits = stripped[1:] if stripped.startswith(("+", "-")) else stripped
        return (
            int(stripped) if digits and digits.isascii() and digits.isdigit() else None
        )
    return None


def _build_step_window(
    start_us: float, duration_us: float, args: Mapping[str, object]
) -> _StepWindow:
    """Construct a `_StepWindow` from microsecond timing and event metadata."""
    phase_raw = args.get("phase")
    return _StepWindow(
        start_us=start_us,
        end_us=start_us + duration_us,
        duration_us=duration_us,
        phase=(
            stripped.lower()
            if type(phase_raw) is str and (stripped := phase_raw.strip())
            else None
        ),
        active_tokens=_int_or_none(args.get("active_tokens")),
        eligible_tokens=_int_or_none(args.get("fused_eligible_tokens")),
    )


def _build_compute_event(
    rank_key: str,
    start_us: float,
    duration_us: float,
    name: str,
    args: Mapping[str, object],
) -> _ComputeEvent:
    """Construct a `_ComputeEvent` from microsecond timing and event metadata."""
    group_size_raw = _int_or_none(args.get("group_size"))
    raw_bytes = args.get("bytes")
    if raw_bytes is None:
        raw_bytes = args.get("byte_size")
    return _ComputeEvent(
        rank_key=rank_key,
        start_us=start_us,
        end_us=start_us + duration_us,
        duration_us=duration_us,
        collective_op=_classify_collective(name),
        group_size=(
            group_size_raw
            if group_size_raw is not None
            and group_size_raw >= MIN_COLLECTIVE_GROUP_SIZE
            else None
        ),
        byte_size=max(0, _int_or_none(raw_bytes) or 0),
    )


def _propagate_perfetto_host_tracks[ItemT](
    tracks_by_rank: dict[str, list[ItemT]],
    events_by_stream: Mapping[str, Sequence[_ComputeEvent]],
) -> None:
    """Attach host-PID wrapper or step tracks to device PIDs lacking local ones."""
    compute_ranks = {
        event.rank_key
        for stream_events in events_by_stream.values()
        for event in stream_events
    }
    if not compute_ranks:
        return
    host_items = [
        item
        for rank_key, items in sorted(tracks_by_rank.items())
        if rank_key not in compute_ranks
        for item in items
    ]
    if not host_items:
        return
    for rank_key in sorted(compute_ranks):
        if not tracks_by_rank.get(rank_key):
            tracks_by_rank[rank_key] = list(host_items)
    for rank_key in list(tracks_by_rank):
        if rank_key not in compute_ranks:
            del tracks_by_rank[rank_key]


def _perfetto_num(value: object, origin_us: int | None = None) -> float | None:
    """Convert a JSON/Decimal scalar to finite float, subtracting `origin_us`."""
    if type(value) is Decimal:
        if not value.is_finite():
            return None
        return float(value - origin_us) if origin_us is not None else float(value)
    if type(value) is int:
        return float(value - origin_us) if origin_us is not None else float(value)
    numeric = as_scalar_number(value)
    if numeric is None:
        return None
    return float(numeric - origin_us) if origin_us is not None else float(numeric)


def _perfetto_axis_label(raw_value: object) -> str:
    """Normalize Perfetto `pid` or `tid` into a rank or stream key component."""
    if (parsed_int := _int_or_none(raw_value)) is not None:
        return str(parsed_int)
    if type(raw_value) is str and (stripped := raw_value.strip()):
        return stripped
    return "0"


def _read_perfetto_events(path: Path) -> tuple[object, ...]:
    """Read, decompress if gzipped, and decode Perfetto trace events."""
    try:
        raw_bytes = path.read_bytes()
        if path.name.endswith(".gz"):
            raw_bytes = gzip.decompress(raw_bytes)
        document: object = json.loads(raw_bytes.decode("utf-8"), parse_float=Decimal)
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"Failed to read Perfetto trace {path}: {exc}"
        raise errors.MegakernelConfigError(msg) from exc
    doc_map = as_mapping(document)
    events_raw = doc_map.get("traceEvents", []) if doc_map is not None else document
    if (event_seq := as_sequence(events_raw)) is None:
        msg = f"Invalid Perfetto trace in {path}: expected traceEvents list"
        raise errors.MegakernelConfigError(msg)
    return event_seq


def _record_perfetto_item(
    item: object,
    origin_us: int,
    parsed: _ParsedTrace,
) -> None:
    """Record one raw Perfetto `X` event into module, step, or compute buckets."""
    item_map = as_mapping(item)
    if item_map is None or item_map.get("ph") != "X":
        return
    start_us = _perfetto_num(item_map.get("ts"), origin_us)
    duration_us = _perfetto_num(item_map.get("dur"))
    if start_us is None or duration_us is None or duration_us <= 0.0:
        return
    name = str(item_map.get("name", ""))
    category = str(item_map.get("cat", ""))
    raw_args = as_mapping(item_map.get("args")) or {}
    args: dict[str, object] = {
        k: float(v) if type(v) is Decimal and v.is_finite() else v
        for k, v in raw_args.items()
    }
    rank_key = f"pid:{_perfetto_axis_label(item_map.get('pid'))}"
    if _is_module_wrapper(name, category, ""):
        parsed.modules_by_rank.setdefault(rank_key, []).append(
            _ModuleWindow(name, start_us, start_us + duration_us)
        )
    elif category.strip().lower() not in PERFETTO_COMPUTE_CATEGORIES and _is_step_event(
        name, category, "", args
    ):
        parsed.steps_by_rank.setdefault(rank_key, []).append(
            _build_step_window(start_us, duration_us, args)
        )
    else:
        stream_key = f"{rank_key}:tid:{_perfetto_axis_label(item_map.get('tid'))}"
        parsed.events_by_stream.setdefault(stream_key, []).append(
            _build_compute_event(rank_key, start_us, duration_us, name, args)
        )


def _parse_perfetto_file(path: Path) -> _ParsedTrace:
    """Parse a `.json` or `.json.gz` Perfetto trace into normalized streams."""
    raw_events = _read_perfetto_events(path)
    valid_ts_ints: list[int] = []
    for item in raw_events:
        item_map = as_mapping(item)
        if item_map is None or item_map.get("ph") != "X":
            continue
        duration_us = _perfetto_num(item_map.get("dur"))
        raw_ts = item_map.get("ts")
        ts_num = _perfetto_num(raw_ts)
        if duration_us is None or duration_us <= 0.0 or ts_num is None:
            continue
        valid_ts_ints.append(
            int(raw_ts)
            if isinstance(raw_ts, Decimal | int) and not isinstance(raw_ts, bool)
            else int(ts_num)
        )
    origin_us = min(valid_ts_ints, default=0)

    parsed = _ParsedTrace(kind="perfetto", file_label=path.name)
    for item in raw_events:
        _record_perfetto_item(item, origin_us, parsed)
    _propagate_perfetto_host_tracks(parsed.steps_by_rank, parsed.events_by_stream)
    _propagate_perfetto_host_tracks(parsed.modules_by_rank, parsed.events_by_stream)
    return parsed


def _decode_xstat_value(
    stat: _XStatProto, stat_meta: Mapping[int, _XStatMetaProto]
) -> object | None:
    """Decode one `XStat.value` oneof member including `bytes_value` and `ref_value`."""
    decoded: object | None = None
    match stat.WhichOneof("value"):
        case "bytes_value":
            raw_bytes = stat.bytes_value
            decoded = (
                raw_bytes.decode("utf-8", errors="replace")
                if isinstance(raw_bytes, bytes | bytearray)
                else str(raw_bytes)
            )
        case "ref_value":
            ref_id = int(stat.ref_value)
            ref_meta = stat_meta.get(ref_id)
            decoded = str(ref_meta.name) if ref_meta is not None else ref_id
        case "str_value":
            decoded = str(stat.str_value)
        case "double_value":
            decoded = float(stat.double_value)
        case "uint64_value":
            decoded = int(stat.uint64_value)
        case "int64_value":
            decoded = int(stat.int64_value)
        case _:
            decoded = None
    return decoded


def _extract_xstats(
    stats: Sequence[_XStatProto], stat_meta: Mapping[int, _XStatMetaProto]
) -> dict[str, object]:
    """Extract a mapping of decoded scalar stats from a repeated `XStat` sequence."""
    stats_dict: dict[str, object] = {}
    for stat in stats:
        meta_entry = stat_meta.get(stat.metadata_id)
        if (
            meta_entry is not None
            and (decoded := _decode_xstat_value(stat, stat_meta)) is not None
        ):
            stats_dict[str(meta_entry.name)] = decoded
    return stats_dict


def _ingest_xplane_line(
    line: _XLineProto,
    context: _PlaneIngestContext,
    parsed: _ParsedTrace,
) -> None:
    """Ingest all timed events on one XLine into module, step, or compute buckets."""
    line_name = str(line.name or line.display_name or "")
    line_lower = line_name.strip().lower()
    line_timestamp_ns = int(line.timestamp_ns)
    # Subtract integer per-plane origin before float conversion to prevent ULP loss.
    line_base_us = (
        float(line_timestamp_ns - context.origin_ns) / NS_PER_US
        if line_timestamp_ns > 0
        else 0.0
    )
    is_compute_line = line_lower in XPLANE_COMPUTE_LINES
    stream_key = f"{context.rank_key}:line:{int(line.id)}:{line_name}"
    for event in line.events:
        if event.WhichOneof("data") == "num_occurrences":
            continue
        duration_us = float(event.duration_ps) / PS_PER_US
        if duration_us <= 0.0:
            continue
        start_us = line_base_us + float(event.offset_ps) / PS_PER_US
        event_meta = context.event_metadata.get(event.metadata_id)
        event_name = (
            str(event_meta.display_name or event_meta.name)
            if event_meta is not None
            else f"ev_{event.metadata_id}"
        )
        base_stats = context.meta_stats.get(event.metadata_id)
        if event.stats:
            args = dict(base_stats) if base_stats else {}
            args.update(_extract_xstats(event.stats, context.stat_metadata))
        else:
            args = base_stats or {}
        if _is_module_wrapper(event_name, "", line_name):
            parsed.modules_by_rank.setdefault(context.rank_key, []).append(
                _ModuleWindow(event_name, start_us, start_us + duration_us)
            )
        elif not is_compute_line and _is_step_event(event_name, "", line_name, args):
            parsed.steps_by_rank.setdefault(context.rank_key, []).append(
                _build_step_window(start_us, duration_us, args)
            )
        elif not context.has_xla_ops_line or is_compute_line:
            parsed.events_by_stream.setdefault(stream_key, []).append(
                _build_compute_event(
                    context.rank_key, start_us, duration_us, event_name, args
                )
            )


def _discover_xplane_files(path: Path) -> list[Path]:
    """Discover `.xplane.pb` files from a single path or profile directory."""
    if path.is_file():
        return [path]
    if merge_xplane is None:
        msg = "merge_xplane or xplane_pb2 tool not available"
        raise XPlaneToolUnavailableError(msg)
    sessions = merge_xplane.discover_sessions([str(path)])
    if len(sessions) > 1:
        dirs = ", ".join(str(session.directory) for session in sessions)
        msg = (
            f"Multiple XPlane sessions found under {path} ({dirs}); "
            "pass a single session directory"
        )
        raise errors.MegakernelConfigError(msg)
    if len(sessions) == 1:
        return [Path(capture.path) for capture in sessions[0].captures]
    return []


def _parse_xplane_target(path: Path) -> _ParsedTrace:
    """Parse one or more `.xplane.pb` files via streaming per-plane origin."""
    if xplane_pb2 is None:
        msg = (
            "XPlane parsing requires vllm_torchtpu.tools.xplane_pb2, "
            "which could not be loaded"
        )
        raise XPlaneToolUnavailableError(msg)
    xplane_pb2_module = cast("_XPlaneModuleProto", xplane_pb2)
    files = _discover_xplane_files(path)
    if not files:
        msg = f"No .xplane.pb files found under {path}"
        raise errors.MegakernelConfigError(msg)

    parsed = _ParsedTrace(kind="xplane", file_label=path.name)
    for file_index, xplane_path in enumerate(files):
        space = xplane_pb2_module.XSpace()
        try:
            space.ParseFromString(xplane_path.read_bytes())
        except (OSError, DecodeError) as exc:
            msg = f"Failed to parse XPlane trace {xplane_path}: {exc}"
            raise errors.MegakernelConfigError(msg) from exc
        for plane in space.planes:
            plane_name = str(plane.name)
            if (
                not plane_name.startswith(XPLANE_DEVICE_PLANE_PREFIX)
                or XPLANE_SPARSE_FILTER in plane_name.lower()
            ):
                continue
            active_lines = [
                line
                for line in plane.lines
                if not any(
                    marker in str(line.name or line.display_name or "").lower()
                    for marker in XPLANE_ANNOTATION_LINE_MARKERS
                )
            ]
            positive_timestamps_ns = [
                int(line.timestamp_ns)
                for line in active_lines
                if line.events and int(line.timestamp_ns) > 0
            ]
            has_xla_ops_line = any(
                str(line.name or line.display_name or "").strip().lower()
                in XPLANE_COMPUTE_LINES
                for line in active_lines
            )
            event_metadata = dict(plane.event_metadata.items())
            stat_metadata = dict(plane.stat_metadata.items())
            meta_stats = {
                meta_id: _extract_xstats(meta_entry.stats, stat_metadata)
                for meta_id, meta_entry in event_metadata.items()
                if meta_entry.stats
            }
            context = _PlaneIngestContext(
                rank_key=f"file:{file_index}:plane:{plane.id}:{plane_name}",
                origin_ns=min(positive_timestamps_ns, default=0),
                has_xla_ops_line=has_xla_ops_line,
                event_metadata=event_metadata,
                stat_metadata=stat_metadata,
                meta_stats=meta_stats,
            )
            for line in active_lines:
                _ingest_xplane_line(line, context, parsed)
    return parsed


def _steady_state_steps(steps: Sequence[_StepWindow]) -> list[_StepWindow]:
    """Sort step windows and drop compilation/warmup step 0 under both thresholds."""
    ordered = sorted(steps, key=attrgetter("start_us"))
    if len(ordered) >= MIN_WARMUP_DROP_STEPS:
        min_steady_duration_us = min(step.duration_us for step in ordered[1:])
        if (
            ordered[0].duration_us > WARMUP_OUTLIER_RATIO * min_steady_duration_us
            or len(ordered) >= MIN_STEPS_FOR_WARMUP_DROP
        ):
            return ordered[1:]
    return ordered


def _scan_event_window(
    window_events: Sequence[_ComputeEvent],
) -> tuple[list[float], list[_ComputeEvent]]:
    """Collect positive non-overlapping gaps via `max_end_us` and collectives."""
    gaps: list[float] = []
    collectives: list[_ComputeEvent] = []
    max_end_us: float | None = None
    for current_event in window_events:
        if current_event.collective_op is not None:
            collectives.append(current_event)
        if max_end_us is not None:
            gap_us = current_event.start_us - max_end_us
            if 0.0 < gap_us <= MAX_INTRA_STEP_GAP_US:
                gaps.append(gap_us)
            max_end_us = max(max_end_us, current_event.end_us)
        else:
            max_end_us = current_event.end_us
    return gaps, collectives


def _collect_stream_gaps_and_collectives(
    stream_events: Sequence[_ComputeEvent],
    steady_steps: Sequence[_StepWindow],
) -> tuple[list[float], list[_ComputeEvent]]:
    """Extract intra-step positive launch gaps and steady-state collective events."""
    ordered = sorted(stream_events, key=attrgetter("start_us"))
    if steady_steps:
        starts = [event.start_us for event in ordered]
        gaps: list[float] = []
        collectives: list[_ComputeEvent] = []
        for step_window in steady_steps:
            start_idx = bisect_left(starts, step_window.start_us)
            end_idx = bisect_left(starts, step_window.end_us)
            window_gaps, window_collectives = _scan_event_window(
                ordered[start_idx:end_idx]
            )
            gaps.extend(window_gaps)
            collectives.extend(window_collectives)
        return gaps, collectives

    trimmed = ordered[1:] if len(ordered) >= MIN_EVENTS_FOR_WARMUP_DROP else ordered
    # No step window: the gaps between device events are not launch gaps.
    _, collectives = _scan_event_window(trimmed)
    return [], collectives


def _summarize_collectives(
    collectives: Sequence[_ComputeEvent],
    default_group_size: int | None = None,
) -> list[CollectiveSummary]:
    """Group collectives by `(op, group_size, bytes)` and compute medians and means."""
    buckets: dict[tuple[str, int, int], list[float]] = {}
    for event in collectives:
        group_size = (
            event.group_size if event.group_size is not None else default_group_size
        )
        if (
            event.collective_op is None
            or group_size is None
            or group_size < MIN_COLLECTIVE_GROUP_SIZE
        ):
            continue
        buckets.setdefault(
            (event.collective_op, group_size, event.byte_size), []
        ).append(event.duration_us)

    return [
        {
            "op": op_name,
            "group_size": group_size,
            "bytes": num_bytes,
            "latency_us": _percentile(sorted_durations, 0.5),
            "mean_latency_us": round(
                sum(sorted_durations) / len(sorted_durations), QUANTILE_DECIMALS
            ),
            "n": len(sorted_durations),
        }
        for (op_name, group_size, num_bytes), durations in sorted(buckets.items())
        for sorted_durations in (sorted(durations),)
    ]


def _summarize_step_coverage(
    steady_steps: Sequence[_StepWindow],
) -> StepCoverageSummary | None:
    """Compute validated `step_coverage`, returning `None` when absent or zero.

    Token-bearing-only traces compute the active-token-weighted fraction
    `sum(eligible) / sum(active)`. Traces that mix in phase-only steps (without
    token counts) fall back to the unweighted mean across all annotated step
    fractions so every annotated step carries equal weight.
    """
    token_active = 0
    token_eligible = 0
    token_fractions: list[float] = []
    phase_fractions: list[float] = []
    for step_window in steady_steps:
        active, eligible = step_window.active_tokens, step_window.eligible_tokens
        if (
            (active is not None and active < 0)
            or (eligible is not None and eligible < 0)
            or (active is not None and eligible is not None and eligible > active)
        ):
            msg = (
                "Invalid step token counts: "
                f"active_tokens={active}, fused_eligible_tokens={eligible}"
            )
            raise errors.MegakernelConfigError(msg)
        if active is not None and eligible is not None:
            if active > 0:
                token_active += active
                token_eligible += eligible
                token_fractions.append(eligible / active)
        elif step_window.phase is not None:
            phase_fractions.append(
                1.0 if step_window.phase in FUSED_ELIGIBLE_PHASES else 0.0
            )
    meta_count = len(token_fractions) + len(phase_fractions)
    if meta_count == 0:
        return None
    fraction = (
        token_eligible / token_active
        if token_fractions and not phase_fractions
        else sum([*token_fractions, *phase_fractions]) / meta_count
    )
    rounded = round(fraction, QUANTILE_DECIMALS)
    if rounded <= 0.0:
        return None
    return {
        "fused_eligible_fraction": rounded,
        "basis": STEP_COVERAGE_BASIS,
        "n": meta_count,
    }


def _load_parsed_trace(trace_path: str | Path) -> _ParsedTrace:
    """Validate `trace_path` and dispatch to the XPlane or Perfetto parser."""
    path = Path(trace_path)
    if not path.exists():
        msg = f"Missing trace path: {path}"
        raise errors.MegakernelConfigError(msg)
    if path.is_dir() or path.name.endswith(".xplane.pb"):
        return _parse_xplane_target(path)
    if path.name.endswith((".json", ".json.gz")):
        return _parse_perfetto_file(path)
    msg = f"Unsupported trace format: {path.name}"
    raise errors.MegakernelConfigError(msg)


def _build_source_meta(
    parsed: _ParsedTrace,
    options: CalibrateOptions,
) -> CalibrationSourceMeta:
    """Assemble provenance metadata for the calibrated output payload."""
    source_meta: CalibrationSourceMeta = {
        "kind": parsed.kind,
        "file": parsed.file_label,
        "chip": options.chip,
        "phase": options.phase,
    }
    if options.model is not None:
        source_meta["model"] = options.model
    if options.batch is not None:
        source_meta["batch"] = options.batch
    if options.tp is not None:
        source_meta["tp"] = options.tp
    return source_meta


def _merged_busy_duration_us(intervals: Sequence[tuple[float, float]]) -> float:
    """Compute union duration across sorted or overlapping `(start, end)` spans."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=itemgetter(0))
    current_start_us, current_end_us = ordered[0]
    total_duration_us = 0.0
    for start_us, end_us in ordered[1:]:
        if start_us <= current_end_us:
            current_end_us = max(current_end_us, end_us)
        else:
            total_duration_us += current_end_us - current_start_us
            current_start_us, current_end_us = start_us, end_us
    return total_duration_us + (current_end_us - current_start_us)


def _group_events_by_rank(
    events_by_stream: Mapping[str, Sequence[_ComputeEvent]],
) -> dict[str, list[_ComputeEvent]]:
    """Aggregate stream-scoped compute events into sorted per-rank lists."""
    events_by_rank: dict[str, list[_ComputeEvent]] = {}
    for stream_events in events_by_stream.values():
        for event in stream_events:
            events_by_rank.setdefault(event.rank_key, []).append(event)
    for rank_events in events_by_rank.values():
        rank_events.sort(key=attrgetter("start_us"))
    return events_by_rank


def _collect_step_idle_gaps(
    events_by_rank: Mapping[str, Sequence[_ComputeEvent]],
    eval_steps_by_rank: Mapping[str, Sequence[_StepWindow]],
) -> tuple[list[float], int]:
    """Measure interval-merged per-step idle gaps and max step op count."""
    step_idle_gaps_us: list[float] = []
    max_step_ops = 0
    for rank_key in sorted(eval_steps_by_rank):
        rank_events = events_by_rank.get(rank_key, ())
        for step_window in eval_steps_by_rank[rank_key]:
            clipped = [
                (
                    max(step_window.start_us, event.start_us),
                    min(step_window.end_us, event.end_us),
                )
                for event in rank_events
                if event.end_us > step_window.start_us
                and event.start_us < step_window.end_us
            ]
            max_step_ops = max(max_step_ops, len(clipped))
            if clipped:
                active_span_us = max(end for _, end in clipped) - min(
                    start for start, _ in clipped
                )
                step_idle_gaps_us.append(
                    max(0.0, active_span_us - _merged_busy_duration_us(clipped))
                )
    return step_idle_gaps_us, max_step_ops


def _compute_conserved_launch_gap(
    parsed: _ParsedTrace,
    events_by_rank: Mapping[str, Sequence[_ComputeEvent]],
    eval_steps_by_rank: Mapping[str, Sequence[_StepWindow]],
    norm_layers: int | None,
    *,
    explicit_norm: bool,
) -> QuantileSummary | None:
    """Normalize interval-merged step idle time by `2 * norm_layers` boundaries."""
    step_idle_gaps_us, max_step_ops = _collect_step_idle_gaps(
        events_by_rank, eval_steps_by_rank
    )
    boundaries = (
        SUBLAYER_BOUNDARIES_PER_LAYER * norm_layers if norm_layers is not None else None
    )
    op_floor = (
        min(CONSERVED_TRACE_MIN_OPS, boundaries)
        if boundaries is not None
        else CONSERVED_TRACE_MIN_OPS
    )
    is_conserved = bool(step_idle_gaps_us) and (
        (
            (bool(parsed.modules_by_rank) or explicit_norm)
            and max_step_ops >= MIN_STEP_OPS_FOR_EXPLICIT_NORM
        )
        or max_step_ops >= op_floor
    )
    if not is_conserved:
        return None
    if norm_layers is None or boundaries is None:
        msg = (
            "Conserved sublayer-boundary launch gap normalization requires "
            "--model or --num-layers (no layer-count default is assumed)"
        )
        raise errors.MegakernelConfigError(msg)
    sorted_idles = sorted(step_idle_gaps_us)
    p50_idle = _percentile(sorted_idles, 0.5, CONSERVED_GAP_DECIMALS)
    p90_idle = _percentile(sorted_idles, 0.9, CONSERVED_GAP_DECIMALS)
    return {
        "p50": round(p50_idle / boundaries, CONSERVED_GAP_DECIMALS),
        "p90": round(p90_idle / boundaries, CONSERVED_GAP_DECIMALS),
        "n": len(sorted_idles),
        "step_idle_gap_us": round(p50_idle, QUANTILE_DECIMALS),
        "num_layers_norm": norm_layers,
        "unit": CONSERVED_GAP_UNIT,
    }


def _filter_phase_steps(
    steady_by_rank: Mapping[str, list[_StepWindow]],
    phase: str,
) -> tuple[dict[str, list[_StepWindow]], list[_StepWindow]]:
    """Filter steady-state step windows by requested serving phase."""
    if phase == "mixed":
        all_steps = [
            step_window
            for rank_key in sorted(steady_by_rank)
            for step_window in steady_by_rank[rank_key]
        ]
        return {
            rank_key: list(steps) for rank_key, steps in steady_by_rank.items()
        }, all_steps

    allowed = FUSED_ELIGIBLE_PHASES if phase == "decode" else (phase,)
    filtered = {
        rank_key: [
            step_window
            for step_window in steps
            if step_window.phase is None or step_window.phase in allowed
        ]
        for rank_key, steps in steady_by_rank.items()
    }
    matched = [step_window for steps in filtered.values() for step_window in steps]
    if not matched and any(steady_by_rank.values()):
        found_phases = {
            step_window.phase
            for steps in steady_by_rank.values()
            for step_window in steps
            if step_window.phase is not None
        }
        msg = (
            f"No steps match requested phase '{phase}'; found phases: "
            f"{sorted(found_phases)}"
        )
        raise errors.MegakernelConfigError(msg)
    return filtered, matched


def _compute_launch_gap_and_collectives(
    parsed: _ParsedTrace,
    events_by_rank: Mapping[str, Sequence[_ComputeEvent]],
    eval_steps_by_rank: Mapping[str, Sequence[_StepWindow]],
    norm_layers: int | None,
    *,
    explicit_norm: bool,
) -> tuple[QuantileSummary | None, list[_ComputeEvent]]:
    """Collect intra-step gaps and collectives and compute launch gap summary."""
    all_gaps: list[float] = []
    all_collectives: list[_ComputeEvent] = []
    for stream_key in sorted(parsed.events_by_stream):
        stream_events = parsed.events_by_stream[stream_key]
        if not stream_events:
            continue
        gaps, collectives = _collect_stream_gaps_and_collectives(
            stream_events, eval_steps_by_rank.get(stream_events[0].rank_key, ())
        )
        all_gaps.extend(gaps)
        all_collectives.extend(collectives)

    conserved_gap = _compute_conserved_launch_gap(
        parsed,
        events_by_rank,
        eval_steps_by_rank,
        norm_layers,
        explicit_norm=explicit_norm,
    )
    if conserved_gap is not None:
        return conserved_gap, all_collectives
    sorted_gaps = sorted(all_gaps)
    if not sorted_gaps:
        return None, all_collectives
    return {
        "p50": _percentile(sorted_gaps, 0.5),
        "p90": _percentile(sorted_gaps, 0.9),
        "n": len(sorted_gaps),
    }, all_collectives


def _trim_module_window(
    module_window: _ModuleWindow,
    rank_events: Sequence[_ComputeEvent],
    starts: Sequence[float],
) -> _StepWindow | None:
    """Trim one XLA Modules wrapper to its inner device ops and record host head."""
    start_idx = bisect_left(starts, module_window.start_us)
    end_idx = bisect_left(starts, module_window.end_us)
    if start_idx >= end_idx:
        return None
    first_op_start_us = rank_events[start_idx].start_us
    last_op_end_us = min(
        module_window.end_us,
        max(event.end_us for event in rank_events[start_idx:end_idx]),
    )
    if last_op_end_us <= first_op_start_us:
        return None
    return _StepWindow(
        start_us=first_op_start_us,
        end_us=last_op_end_us,
        duration_us=last_op_end_us - first_op_start_us,
        host_head_us=max(0.0, first_op_start_us - module_window.start_us),
    )


def _trim_rank_module_programs(
    module_windows: Sequence[_ModuleWindow],
    rank_events: Sequence[_ComputeEvent],
) -> list[_StepWindow]:
    """Select the primary XLA Modules program by steady duration and trim to ops."""
    if not module_windows or not rank_events:
        return []
    starts = [event.start_us for event in rank_events]
    by_program: dict[str, list[_StepWindow]] = {}
    for module_window in module_windows:
        if (
            trimmed := _trim_module_window(module_window, rank_events, starts)
        ) is not None:
            by_program.setdefault(module_window.program_name, []).append(trimmed)
    if not by_program:
        return []

    def _program_key(item: tuple[str, list[_StepWindow]]) -> tuple[float, str]:
        """Rank programs by descending steady-state duration, then name."""
        program_name, windows = item
        steady = _steady_state_steps(windows)
        return (-sum(window.duration_us for window in steady), program_name)

    return sorted(
        min(by_program.items(), key=_program_key)[1], key=attrgetter("start_us")
    )


def _collect_eval_host_heads(
    parsed: _ParsedTrace,
    eval_steps_by_rank: Mapping[str, Sequence[_StepWindow]],
    events_by_rank: Mapping[str, Sequence[_ComputeEvent]],
) -> QuantileSummary | None:
    """Summarize pre-first-op host heads aligned to surviving phase steps."""
    steady_heads: list[float] = []
    for rank_key, eval_steps in sorted(eval_steps_by_rank.items()):
        if not eval_steps:
            continue
        embedded = [
            step.host_head_us for step in eval_steps if step.host_head_us is not None
        ]
        if embedded:
            steady_heads.extend(embedded)
            continue
        module_windows = parsed.modules_by_rank.get(rank_key)
        rank_events = events_by_rank.get(rank_key, ())
        if not module_windows or not rank_events:
            continue
        starts = [event.start_us for event in rank_events]
        trimmed_windows = [
            trimmed
            for module_window in module_windows
            if (trimmed := _trim_module_window(module_window, rank_events, starts))
            is not None
        ]
        for step_window in eval_steps:
            overlapping = [
                (
                    min(step_window.end_us, module_window.end_us)
                    - max(step_window.start_us, module_window.start_us),
                    module_window.host_head_us,
                )
                for module_window in trimmed_windows
                if module_window.host_head_us is not None
            ]
            best_overlap, best_head = max(overlapping, default=(0.0, None))
            if best_overlap > 0.0 and best_head is not None:
                steady_heads.append(best_head)
    if not steady_heads:
        return None
    sorted_heads = sorted(steady_heads)
    return {
        "p50": _percentile(sorted_heads, 0.5),
        "p90": _percentile(sorted_heads, 0.9),
        "n": len(sorted_heads),
    }


def calibrate_trace(
    trace_path: str | Path,
    out_path: str | Path | None = None,
    *,
    options: CalibrateOptions | None = None,
) -> CalibrationPayload:
    """Calibrate advisor priors from an XPlane or Perfetto trace file or directory.

    Returns:
        Validated `advisor.measurements.v1` dictionary (`CalibrationPayload`).
    """
    calibration_options = options or CalibrateOptions()
    resolve_chip_spec(calibration_options.chip)
    model_profile = (
        resolve_model_profile(calibration_options.model)
        if calibration_options.model is not None
        else None
    )
    norm_layers = (
        calibration_options.num_layers
        if calibration_options.num_layers is not None
        else (model_profile.num_layers if model_profile is not None else None)
    )
    parsed = _load_parsed_trace(trace_path)
    events_by_rank = _group_events_by_rank(parsed.events_by_stream)
    for rank_key, module_windows in sorted(parsed.modules_by_rank.items()):
        if not parsed.steps_by_rank.get(rank_key) and (
            trimmed := _trim_rank_module_programs(
                module_windows, events_by_rank.get(rank_key, ())
            )
        ):
            parsed.steps_by_rank[rank_key] = trimmed
    steady_by_rank = {
        rank_key: _steady_state_steps(steps)
        for rank_key, steps in parsed.steps_by_rank.items()
    }
    eval_steps_by_rank, eval_steady_steps = _filter_phase_steps(
        steady_by_rank, calibration_options.phase
    )
    launch_gap, all_collectives = _compute_launch_gap_and_collectives(
        parsed,
        events_by_rank,
        eval_steps_by_rank,
        norm_layers,
        explicit_norm=calibration_options.num_layers is not None,
    )
    payload: CalibrationPayload = {
        "schema": MEASUREMENTS_SCHEMA,
        "source": _build_source_meta(parsed, calibration_options),
        "collectives": _summarize_collectives(all_collectives, calibration_options.tp),
    }
    if launch_gap is not None:
        payload["launch_gap_us"] = launch_gap
    if (
        step_coverage := _summarize_step_coverage(
            [
                step_window
                for rank_key in sorted(steady_by_rank)
                for step_window in steady_by_rank[rank_key]
            ]
        )
    ) is not None:
        payload["step_coverage"] = step_coverage
    if eval_steady_steps:
        step_durations = sorted(step.duration_us for step in eval_steady_steps)
        payload["step_time_us"] = {
            "p50": _positive_percentile(step_durations, 0.5),
            "p90": _positive_percentile(step_durations, 0.9),
            "n": len(step_durations),
        }
    if (
        host_head := _collect_eval_host_heads(
            parsed, eval_steps_by_rank, events_by_rank
        )
    ) is not None:
        payload["host_head_us"] = host_head
    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
