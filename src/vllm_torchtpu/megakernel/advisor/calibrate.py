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
import math
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from vllm_torchtpu.megakernel.core import errors

from .constants import (
    COLLECTIVE_IGNORE_SUFFIXES,
    COLLECTIVE_PREFIXES,
    CONSERVED_GAP_DECIMALS,
    CONSERVED_GAP_UNIT,
    DEFAULT_CALIBRATE_LAYERS,
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

try:
    from google.protobuf.message import DecodeError

    from vllm_torchtpu.tools import merge_xplane, xplane_pb2
except (ImportError, RuntimeError):
    DecodeError = ValueError
    merge_xplane = None
    xplane_pb2 = None


class XPlaneToolUnavailableError(errors.MegakernelConfigError):
    """Raised when xplane tools (merge_xplane or xplane_pb2) are unavailable."""

    def __init__(
        self,
        message: str = "merge_xplane or xplane_pb2 tool not available",
    ) -> None:
        """Initialize XPlaneToolUnavailableError with a default message."""
        super().__init__(message)


if TYPE_CHECKING:
    from typing import Protocol

    class _XStatProto(Protocol):
        """Structural protocol for an XStat protobuf message."""

        metadata_id: int
        int64_value: int
        uint64_value: int
        double_value: float
        str_value: str

    class _XStatMetaProto(Protocol):
        """Structural protocol for an XStatMetadata protobuf message."""

        id: int
        name: str

    class _XEventMetaProto(Protocol):
        """Structural protocol for an XEventMetadata protobuf message."""

        id: int
        name: str
        display_name: str

    class _XEventProto(Protocol):
        """Structural protocol for an XEvent protobuf message."""

        metadata_id: int
        offset_ps: int
        duration_ps: int
        stats: Sequence[_XStatProto]

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


@dataclass(frozen=True)
class CalibrateOptions:
    """Typed calibration provenance and layer-normalization options."""

    chip: str = DEFAULT_CHIP
    model: str | None = None
    phase: str = DEFAULT_PHASE
    batch: int | None = None
    tp: int | None = None
    num_layers: int | None = None


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
    hop_latency_us: float
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


@dataclass(frozen=True)
class _StepWindow:
    """Normalized execution step interval with optional token-eligibility counts."""

    start_us: float
    end_us: float
    duration_us: float
    phase: str | None = None
    active_tokens: int | None = None
    eligible_tokens: int | None = None


@dataclass(frozen=True)
class _ComputeEvent:
    """Normalized compute or collective trace event scoped to a rank stream."""

    rank_key: str
    stream_key: str
    start_us: float
    end_us: float
    duration_us: float
    collective_op: str | None = None
    group_size: int | None = None
    byte_size: int = 0


@dataclass(frozen=True)
class _ParsedTrace:
    """Intermediate normalized representation of an XPlane or Perfetto trace."""

    kind: str
    file_label: str
    steps_by_rank: dict[str, list[_StepWindow]]
    events_by_stream: dict[str, list[_ComputeEvent]]


def _percentile_prec(
    sorted_vals: Sequence[float], q: float, ndigits: int = QUANTILE_DECIMALS
) -> float:
    """Compute linear-interpolated quantile `q` rounded to `ndigits` decimals."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, ndigits)


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Compute linear-interpolated quantile `q` in `[0, 1]` from sorted floats."""
    return _percentile_prec(sorted_vals, q, QUANTILE_DECIMALS)


def _classify_collective(name: str) -> str | None:
    """Return canonical collective op name if `name` is a collective event."""
    norm = name.lower().replace("_", "-")
    if any(suffix in norm for suffix in COLLECTIVE_IGNORE_SUFFIXES):
        return None
    for prefix, canonical in COLLECTIVE_PREFIXES:
        if prefix in norm:
            return canonical
    return None


def _is_module_wrapper(name: str, category: str, line_name: str) -> bool:
    """Return True when an event or line wraps full-step XLA modules."""
    low_category = category.lower()
    low_line = line_name.lower()
    low_name = name.lower()
    return any(
        marker in low_line or marker in low_category for marker in XLA_MODULE_MARKERS
    ) or low_name.startswith("xla_module")


def _is_step_event(
    name: str, category: str, line_name: str, args: Mapping[str, object]
) -> bool:
    """Return True when an event marks a model execution step window."""
    low_name = name.lower()
    return (
        "step" in category.lower()
        or "step" in line_name.lower()
        or low_name.startswith("step")
        or low_name == "model_step"
        or "step" in args
        or "step_num" in args
    )


def _int_or_none(value: object) -> int | None:
    """Convert a JSON/proto integer-like scalar (excluding bool) to int."""
    match value:
        case bool():
            return None
        case int():
            return value
        case float():
            return int(value) if math.isfinite(value) else None
        case str():
            stripped = value.strip()
            return int(stripped) if stripped.isdigit() else None
        case _:
            return None


def _finite_float_or_none(value: object) -> float | None:
    """Convert an int/float scalar (excluding bool) to a finite float."""
    match value:
        case bool():
            return None
        case int() | float():
            result = float(value)
            return result if math.isfinite(result) else None
        case _:
            return None


def _build_step_window(
    start_us: float, duration_us: float, args: Mapping[str, object]
) -> _StepWindow:
    """Construct a `_StepWindow` from microsecond timing and event metadata."""
    match args.get("phase"):
        case str() as phase_raw if phase_raw:
            phase_str = phase_raw.lower()
        case _:
            phase_str = None
    active_tokens = _int_or_none(args.get("active_tokens"))
    eligible_tokens = _int_or_none(args.get("fused_eligible_tokens"))
    return _StepWindow(
        start_us=start_us,
        end_us=start_us + duration_us,
        duration_us=duration_us,
        phase=phase_str,
        active_tokens=active_tokens,
        eligible_tokens=eligible_tokens,
    )


def _build_compute_event(
    stream_ids: tuple[str, str],
    start_us: float,
    duration_us: float,
    name: str,
    args: Mapping[str, object],
) -> _ComputeEvent:
    """Construct a `_ComputeEvent` from microsecond timing and event metadata."""
    rank_key, stream_key = stream_ids
    collective_op = _classify_collective(name)
    group_size_raw = _int_or_none(args.get("group_size"))
    group_size = (
        group_size_raw
        if group_size_raw is not None and group_size_raw >= MIN_COLLECTIVE_GROUP_SIZE
        else None
    )
    raw_bytes = args.get("bytes")
    if raw_bytes is None:
        raw_bytes = args.get("byte_size")
    byte_size = _int_or_none(raw_bytes) or 0
    return _ComputeEvent(
        rank_key=rank_key,
        stream_key=stream_key,
        start_us=start_us,
        end_us=start_us + duration_us,
        duration_us=duration_us,
        collective_op=collective_op,
        group_size=group_size,
        byte_size=max(0, byte_size),
    )


def _propagate_perfetto_host_steps(
    steps_by_rank: dict[str, list[_StepWindow]],
    events_by_stream: Mapping[str, Sequence[_ComputeEvent]],
) -> None:
    """Attach host-PID step markers to device PIDs that lack local step tracks."""
    compute_ranks = {
        event.rank_key
        for stream_events in events_by_stream.values()
        for event in stream_events
    }
    if not compute_ranks:
        return
    host_steps = [
        step_window
        for rank_key, steps in sorted(steps_by_rank.items())
        if rank_key not in compute_ranks
        for step_window in steps
    ]
    if not host_steps:
        return
    for rank_key in sorted(compute_ranks):
        if not steps_by_rank.get(rank_key):
            steps_by_rank[rank_key] = list(host_steps)
    for rank_key in list(steps_by_rank):
        if rank_key not in compute_ranks:
            del steps_by_rank[rank_key]


def _read_perfetto_events(path: Path) -> list[object]:
    """Read, decompress if gzipped, and decode a Perfetto trace event list."""
    try:
        raw_bytes = path.read_bytes()
        if path.name.endswith(".gz"):
            raw_bytes = gzip.decompress(raw_bytes)
        text = raw_bytes.decode("utf-8")
        document = json.loads(text)
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"Failed to read Perfetto trace {path}: {exc}"
        raise errors.MegakernelConfigError(msg) from exc
    match document:
        case Mapping() as doc_map:
            events_raw = doc_map.get("traceEvents", [])
        case _:
            events_raw = document
    match events_raw:
        case list() as event_list:
            return event_list
        case _:
            msg = f"Invalid Perfetto trace in {path}: expected traceEvents list"
            raise errors.MegakernelConfigError(msg)


def _record_perfetto_item(
    item: object,
    steps_by_rank: dict[str, list[_StepWindow]],
    events_by_stream: dict[str, list[_ComputeEvent]],
) -> None:
    """Record one raw Perfetto `X` event into step or compute stream buckets."""
    match item:
        case Mapping() as item_map if item_map.get("ph") == "X":
            pass
        case _:
            return
    start_us = _finite_float_or_none(item_map.get("ts"))
    duration_us = _finite_float_or_none(item_map.get("dur"))
    if start_us is None or duration_us is None or duration_us <= 0.0:
        return
    name = str(item_map.get("name", ""))
    category = str(item_map.get("cat", ""))
    process_id = _int_or_none(item_map.get("pid")) or 0
    thread_id = _int_or_none(item_map.get("tid")) or 0
    match item_map.get("args"):
        case Mapping() as args_map:
            args: Mapping[str, object] = args_map
        case _:
            args = {}
    rank_key = f"pid:{process_id}"
    if _is_step_event(name, category, "", args):
        steps_by_rank.setdefault(rank_key, []).append(
            _build_step_window(start_us, duration_us, args)
        )
    elif not _is_module_wrapper(name, category, ""):
        stream_key = f"{rank_key}:tid:{thread_id}"
        events_by_stream.setdefault(stream_key, []).append(
            _build_compute_event(
                (rank_key, stream_key), start_us, duration_us, name, args
            )
        )


def _parse_perfetto_file(path: Path) -> _ParsedTrace:
    """Parse a `.json` or `.json.gz` Perfetto trace into normalized streams."""
    steps_by_rank: dict[str, list[_StepWindow]] = {}
    events_by_stream: dict[str, list[_ComputeEvent]] = {}
    for item in _read_perfetto_events(path):
        _record_perfetto_item(item, steps_by_rank, events_by_stream)
    _propagate_perfetto_host_steps(steps_by_rank, events_by_stream)
    return _ParsedTrace(
        kind="perfetto",
        file_label=path.name,
        steps_by_rank=steps_by_rank,
        events_by_stream=events_by_stream,
    )


def _extract_xevent_stats(
    event: _XEventProto, stat_meta: Mapping[int, _XStatMetaProto]
) -> dict[str, object]:
    """Extract a dictionary of stat values from an XEvent proto."""
    stats_dict: dict[str, object] = {}
    for stat in event.stats:
        meta = stat_meta.get(stat.metadata_id)
        if meta is None:
            continue
        stat_name = str(meta.name)
        if stat.str_value:
            stats_dict[stat_name] = str(stat.str_value)
        elif stat.double_value != 0.0:
            stats_dict[stat_name] = float(stat.double_value)
        elif stat.uint64_value != 0:
            stats_dict[stat_name] = int(stat.uint64_value)
        else:
            stats_dict[stat_name] = int(stat.int64_value)
    return stats_dict


def _ingest_xplane(
    plane: _XPlaneProto,
    rank_key: str,
    steps_by_rank: dict[str, list[_StepWindow]],
    events_by_stream: dict[str, list[_ComputeEvent]],
) -> None:
    """Ingest all lines from a device XPlane into step or compute streams."""
    event_metadata_map = dict(plane.event_metadata.items())
    stat_metadata_map = dict(plane.stat_metadata.items())
    for line in plane.lines:
        line_name = str(line.name or line.display_name or "")
        line_id = int(line.id)
        line_base_us = float(line.timestamp_ns) / NS_PER_US
        stream_key = f"{rank_key}:line:{line_id}:{line_name}"
        for event in line.events:
            duration_us = float(event.duration_ps) / PS_PER_US
            if duration_us <= 0.0:
                continue
            start_us = line_base_us + float(event.offset_ps) / PS_PER_US
            meta = event_metadata_map.get(event.metadata_id)
            event_name = (
                str(meta.display_name or meta.name)
                if meta is not None
                else f"ev_{event.metadata_id}"
            )
            args = _extract_xevent_stats(event, stat_metadata_map)
            if _is_step_event(event_name, "", line_name, args):
                steps_by_rank.setdefault(rank_key, []).append(
                    _build_step_window(start_us, duration_us, args)
                )
            elif not _is_module_wrapper(event_name, "", line_name):
                events_by_stream.setdefault(stream_key, []).append(
                    _build_compute_event(
                        (rank_key, stream_key), start_us, duration_us, event_name, args
                    )
                )


def _discover_xplane_files(path: Path) -> list[Path]:
    """Discover `.xplane.pb` files from a single path or profile directory."""
    if path.is_file():
        return [path]
    if merge_xplane is None:
        raise XPlaneToolUnavailableError
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
    return sorted(path.rglob("*.xplane.pb"))


def _parse_xplane_target(path: Path) -> _ParsedTrace:
    """Parse one or more `.xplane.pb` files using `vllm_torchtpu.tools.xplane_pb2`."""
    if xplane_pb2 is None:
        msg = (
            "XPlane parsing requires vllm_torchtpu.tools.xplane_pb2, "
            "which could not be loaded"
        )
        raise errors.MegakernelConfigError(msg)
    proto_module = cast("_XPlaneModuleProto", xplane_pb2)
    files = _discover_xplane_files(path)
    if not files:
        msg = f"No .xplane.pb files found under {path}"
        raise errors.MegakernelConfigError(msg)

    steps_by_rank: dict[str, list[_StepWindow]] = {}
    events_by_stream: dict[str, list[_ComputeEvent]] = {}
    for file_idx, pb_path in enumerate(files):
        space = proto_module.XSpace()
        try:
            space.ParseFromString(pb_path.read_bytes())
        except (OSError, DecodeError) as exc:
            msg = f"Failed to parse XPlane trace {pb_path}: {exc}"
            raise errors.MegakernelConfigError(msg) from exc
        for plane in space.planes:
            plane_name = str(plane.name)
            if (
                not plane_name.startswith(XPLANE_DEVICE_PLANE_PREFIX)
                or XPLANE_SPARSE_FILTER in plane_name.lower()
            ):
                continue
            rank_key = f"file:{file_idx}:plane:{plane.id}:{plane_name}"
            _ingest_xplane(plane, rank_key, steps_by_rank, events_by_stream)
    return _ParsedTrace(
        kind="xplane",
        file_label=path.name,
        steps_by_rank=steps_by_rank,
        events_by_stream=events_by_stream,
    )


def _steady_state_steps(steps: Sequence[_StepWindow]) -> list[_StepWindow]:
    """Sort step windows and drop compilation/warmup step 0 under both thresholds.

    Drops step 0 when at least `MIN_STEPS_FOR_WARMUP_DROP` (3) steps exist, or when
    at least `MIN_WARMUP_DROP_STEPS` (2) steps exist and step 0 exceeds
    `WARMUP_OUTLIER_RATIO` times the minimum duration of subsequent steps.
    """
    ordered = sorted(steps, key=lambda step_window: step_window.start_us)
    if len(ordered) >= MIN_WARMUP_DROP_STEPS:
        min_steady_step_dur_us = min(
            step_window.duration_us for step_window in ordered[1:]
        )
        if ordered[0].duration_us > WARMUP_OUTLIER_RATIO * min_steady_step_dur_us:
            return list(ordered[1:])
        if len(ordered) >= MIN_STEPS_FOR_WARMUP_DROP:
            return list(ordered[1:])
    return list(ordered)


def _scan_event_window(
    window_events: Sequence[_ComputeEvent],
) -> tuple[list[float], list[_ComputeEvent]]:
    """Collect positive consecutive gaps and collectives within one event window."""
    gaps: list[float] = []
    collectives: list[_ComputeEvent] = []
    for event_index, current_event in enumerate(window_events):
        if current_event.collective_op is not None:
            collectives.append(current_event)
        if event_index > 0:
            gap_us = current_event.start_us - window_events[event_index - 1].end_us
            if 0.0 < gap_us <= MAX_INTRA_STEP_GAP_US:
                gaps.append(gap_us)
    return gaps, collectives


def _collect_stream_gaps_and_collectives(
    stream_events: Sequence[_ComputeEvent],
    steady_steps: Sequence[_StepWindow],
) -> tuple[list[float], list[_ComputeEvent]]:
    """Extract intra-step positive launch gaps and steady-state collective events."""
    ordered = sorted(stream_events, key=lambda event: event.start_us)
    if steady_steps:
        starts = [event.start_us for event in ordered]
        gaps: list[float] = []
        collectives: list[_ComputeEvent] = []
        for step_window in steady_steps:
            lo = bisect_left(starts, step_window.start_us)
            hi = bisect_left(starts, step_window.end_us)
            window_gaps, window_collectives = _scan_event_window(ordered[lo:hi])
            gaps.extend(window_gaps)
            collectives.extend(window_collectives)
        return gaps, collectives

    trimmed = ordered[1:] if len(ordered) >= MIN_EVENTS_FOR_WARMUP_DROP else ordered
    return _scan_event_window(trimmed)


def _summarize_collectives(
    collectives: Sequence[_ComputeEvent],
    default_group_size: int | None = None,
) -> list[CollectiveSummary]:
    """Group collective events by `(op, group_size, bytes)` and compute medians."""
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
        bucket_key = (event.collective_op, group_size, event.byte_size)
        buckets.setdefault(bucket_key, []).append(event.duration_us)

    rows: list[CollectiveSummary] = []
    for (op_name, group_size, num_bytes), durations in sorted(buckets.items()):
        sorted_durations = sorted(durations)
        median_us = _percentile(sorted_durations, 0.5)
        hops = max(1, round(log2(max(MIN_COLLECTIVE_GROUP_SIZE, group_size))))
        rows.append(
            {
                "op": op_name,
                "group_size": group_size,
                "bytes": num_bytes,
                "latency_us": median_us,
                "hop_latency_us": round(median_us / hops, QUANTILE_DECIMALS),
                "n": len(sorted_durations),
            }
        )
    return rows


def _summarize_step_coverage(
    steady_steps: Sequence[_StepWindow],
) -> StepCoverageSummary | None:
    """Compute `step_coverage` when step token or phase metadata is present."""
    total_active = 0
    total_eligible = 0
    meta_count = 0
    for step_window in steady_steps:
        if (
            step_window.active_tokens is not None
            and step_window.eligible_tokens is not None
            and step_window.active_tokens > 0
        ):
            total_active += step_window.active_tokens
            total_eligible += step_window.eligible_tokens
            meta_count += 1
        elif step_window.phase is not None:
            total_active += 1
            total_eligible += 1 if step_window.phase in FUSED_ELIGIBLE_PHASES else 0
            meta_count += 1
    if meta_count == 0 or total_active == 0:
        return None
    return {
        "fused_eligible_fraction": round(
            total_eligible / total_active, QUANTILE_DECIMALS
        ),
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


def _merged_busy_duration_us(
    intervals: Sequence[tuple[float, float]],
) -> float:
    """Compute union duration across sorted or overlapping `(start, end)` spans."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=lambda pair: pair[0])
    current_start_us, current_end_us = ordered[0]
    total_duration_us = 0.0
    for start_us, end_us in ordered[1:]:
        if start_us <= current_end_us:
            current_end_us = max(current_end_us, end_us)
        else:
            total_duration_us += current_end_us - current_start_us
            current_start_us, current_end_us = start_us, end_us
    return total_duration_us + (current_end_us - current_start_us)


def _resolve_norm_layers(model: str | None, explicit_layers: int | None) -> int:
    """Resolve layer count for sublayer boundary normalization."""
    if explicit_layers is not None and explicit_layers > 0:
        return explicit_layers
    if model is not None:
        return resolve_model_profile(model).num_layers
    return DEFAULT_CALIBRATE_LAYERS


def _compute_conserved_launch_gap(
    events_by_stream: Mapping[str, Sequence[_ComputeEvent]],
    eval_steps_by_rank: Mapping[str, Sequence[_StepWindow]],
    norm_layers: int,
    *,
    explicit_norm: bool,
) -> QuantileSummary | None:
    """Normalize interval-merged step idle time by `2 * norm_layers` boundaries."""
    events_by_rank: dict[str, list[_ComputeEvent]] = {}
    for stream_events in events_by_stream.values():
        for event in stream_events:
            events_by_rank.setdefault(event.rank_key, []).append(event)

    boundaries = SUBLAYER_BOUNDARIES_PER_LAYER * norm_layers
    step_idles: list[float] = []
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
                busy_us = _merged_busy_duration_us(clipped)
                step_idles.append(max(0.0, step_window.duration_us - busy_us))

    use_conserved = bool(step_idles) and (
        (explicit_norm and max_step_ops >= MIN_STEP_OPS_FOR_EXPLICIT_NORM)
        or max_step_ops >= boundaries
    )
    if not use_conserved:
        return None
    sorted_idles = sorted(step_idles)
    p50_idle = _percentile_prec(sorted_idles, 0.5, CONSERVED_GAP_DECIMALS)
    p90_idle = _percentile_prec(sorted_idles, 0.9, CONSERVED_GAP_DECIMALS)
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
    eval_steps_by_rank: Mapping[str, Sequence[_StepWindow]],
    options: CalibrateOptions,
) -> tuple[QuantileSummary, list[_ComputeEvent]]:
    """Collect intra-step gaps and collective events and compute launch gap summary."""
    all_gaps: list[float] = []
    all_collectives: list[_ComputeEvent] = []
    for stream_key in sorted(parsed.events_by_stream):
        stream_events = parsed.events_by_stream[stream_key]
        if not stream_events:
            continue
        rank_key = stream_events[0].rank_key
        gaps, collectives = _collect_stream_gaps_and_collectives(
            stream_events, eval_steps_by_rank.get(rank_key, ())
        )
        all_gaps.extend(gaps)
        all_collectives.extend(collectives)

    norm_layers = _resolve_norm_layers(options.model, options.num_layers)
    conserved_gap = _compute_conserved_launch_gap(
        parsed.events_by_stream,
        eval_steps_by_rank,
        norm_layers,
        explicit_norm=options.num_layers is not None,
    )
    sorted_gaps = sorted(all_gaps)
    launch_gap: QuantileSummary = conserved_gap or (
        {
            "p50": _percentile(sorted_gaps, 0.5),
            "p90": _percentile(sorted_gaps, 0.9),
            "n": len(sorted_gaps),
        }
        if sorted_gaps
        else {"n": 0}
    )
    return launch_gap, all_collectives


def calibrate_trace(
    trace_path: str | Path,
    out_path: str | Path | None = None,
    *,
    options: CalibrateOptions | None = None,
) -> CalibrationPayload:
    """Calibrate advisor priors from an XPlane or Perfetto trace file or directory."""
    calibration_options = options or CalibrateOptions()
    parsed = _load_parsed_trace(trace_path)
    steady_by_rank = {
        rank_key: _steady_state_steps(steps)
        for rank_key, steps in parsed.steps_by_rank.items()
    }
    all_steady_steps = [
        step_window
        for rank_key in sorted(steady_by_rank)
        for step_window in steady_by_rank[rank_key]
    ]
    eval_steps_by_rank, eval_steady_steps = _filter_phase_steps(
        steady_by_rank, calibration_options.phase
    )
    launch_gap, all_collectives = _compute_launch_gap_and_collectives(
        parsed, eval_steps_by_rank, calibration_options
    )
    payload: CalibrationPayload = {
        "schema": MEASUREMENTS_SCHEMA,
        "source": _build_source_meta(parsed, calibration_options),
        "launch_gap_us": launch_gap,
        "collectives": _summarize_collectives(all_collectives, calibration_options.tp),
    }
    step_coverage = _summarize_step_coverage(all_steady_steps)
    if step_coverage is not None:
        payload["step_coverage"] = step_coverage
    if eval_steady_steps:
        step_durations = sorted(
            step_window.duration_us for step_window in eval_steady_steps
        )
        payload["step_time_us"] = {
            "p50": _percentile(step_durations, 0.5),
            "p90": _percentile(step_durations, 0.9),
            "n": len(step_durations),
        }
    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
