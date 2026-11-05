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
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from ...tools import merge_xplane, xplane_pb2
from .. import errors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
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


class QuantileSummary(TypedDict, total=False):
    """Summary statistics (`p50`, `p90`, sample count `n`) for a latency series."""

    p50: float
    p90: float
    n: int


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


_MEASUREMENTS_SCHEMA = "advisor.measurements.v1"
_PS_PER_US = 1_000_000.0
_NS_PER_US = 1_000.0
# Gaps above 500 us reflect idle pauses rather than kernel launch overhead.
_MAX_INTRA_STEP_GAP_US = 500.0
_MIN_COLLECTIVE_GROUP_SIZE = 2
_MIN_STEPS_FOR_WARMUP_DROP = 3
_WARMUP_OUTLIER_RATIO: float = 2.0
_MIN_EVENTS_FOR_WARMUP_DROP = 4
_COLLECTIVE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("all-reduce", "all-reduce"),
    ("allreduce", "all-reduce"),
    ("all-gather", "all-gather"),
    ("allgather", "all-gather"),
    ("reduce-scatter", "reduce-scatter"),
    ("reducescatter", "reduce-scatter"),
    ("all-to-all", "all-to-all"),
    ("alltoall", "all-to-all"),
)


@dataclass(frozen=True)
class _StepWindow:
    """Single profiled step interval and optional token/phase metadata."""

    start_us: float
    end_us: float
    dur_us: float
    phase: str | None = None
    active_tokens: int | None = None
    eligible_tokens: int | None = None


@dataclass(frozen=True)
class _ComputeEvent:
    """Single device compute-stream event in microseconds."""

    rank_key: str
    stream_key: str
    start_us: float
    end_us: float
    dur_us: float
    collective_op: str | None = None
    group_size: int | None = None
    byte_size: int = 0


@dataclass(frozen=True)
class _ParsedTrace:
    """Normalized trace streams and step windows extracted from XPlane or Perfetto."""

    kind: str
    file_label: str
    steps_by_rank: dict[str, list[_StepWindow]]
    events_by_stream: dict[str, list[_ComputeEvent]]


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Compute linear-interpolated quantile `q` in `[0, 1]` from sorted floats."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, 4)


def _classify_collective(name: str) -> str | None:
    """Return canonical collective op name if `name` is a collective event."""
    norm = name.lower().replace("_", "-")
    if "-done" in norm or ".done" in norm:
        return None
    for prefix, canonical in _COLLECTIVE_PREFIXES:
        if prefix in norm:
            return canonical
    return None


def _is_step_event(
    name: str, cat: str, line_name: str, args: Mapping[str, object]
) -> bool:
    """Return True when an event marks a model execution step window."""
    low_name = name.lower()
    return (
        "step" in cat.lower()
        or "step" in line_name.lower()
        or low_name.startswith("step")
        or low_name == "model_step"
        or "step" in args
        or "step_num" in args
    )


def _int_or_none(val: object) -> int | None:
    """Convert a primitive scalar to int or return None."""
    if type(val) in (int, float):
        return int(float(str(val)))
    if type(val) is str and val.strip().isdigit():
        return int(val.strip())
    return None


def _build_step_window(
    start_us: float, dur_us: float, args: Mapping[str, object]
) -> _StepWindow:
    """Construct a `_StepWindow` from microsecond timing and event metadata."""
    phase_raw = args.get("phase")
    phase_str = str(phase_raw).lower() if type(phase_raw) is str and phase_raw else None
    act_tok = _int_or_none(args.get("active_tokens"))
    elig_tok = _int_or_none(args.get("fused_eligible_tokens"))
    return _StepWindow(
        start_us=start_us,
        end_us=start_us + dur_us,
        dur_us=dur_us,
        phase=phase_str,
        active_tokens=act_tok,
        eligible_tokens=elig_tok,
    )


def _build_compute_event(
    stream_ids: tuple[str, str],
    start_us: float,
    dur_us: float,
    name: str,
    args: Mapping[str, object],
) -> _ComputeEvent:
    """Construct a `_ComputeEvent` from microsecond timing and event metadata."""
    rank_key, stream_key = stream_ids
    coll_op = _classify_collective(name)
    grp_raw = _int_or_none(args.get("group_size"))
    grp = (
        grp_raw
        if grp_raw is not None and grp_raw >= _MIN_COLLECTIVE_GROUP_SIZE
        else None
    )
    nbytes = _int_or_none(args.get("bytes") or args.get("byte_size")) or 0
    return _ComputeEvent(
        rank_key=rank_key,
        stream_key=stream_key,
        start_us=start_us,
        end_us=start_us + dur_us,
        dur_us=dur_us,
        collective_op=coll_op,
        group_size=grp,
        byte_size=max(0, nbytes),
    )


def _parse_perfetto_file(path: Path) -> _ParsedTrace:
    """Parse a `.json` or `.json.gz` Perfetto trace into normalized streams."""
    raw_bytes = path.read_bytes()
    if path.name.endswith(".gz"):
        text = gzip.decompress(raw_bytes).decode("utf-8")
    else:
        text = raw_bytes.decode("utf-8")
    doc = json.loads(text)
    events_raw = doc.get("traceEvents", []) if type(doc) is dict else doc
    if type(events_raw) is not list:
        msg = f"Invalid Perfetto trace in {path}: expected traceEvents list"
        raise errors.MegakernelConfigError(msg)

    steps_by_rank: dict[str, list[_StepWindow]] = {}
    events_by_stream: dict[str, list[_ComputeEvent]] = {}
    for item in events_raw:
        if type(item) is not dict or item.get("ph") != "X":
            continue
        ts_val = item.get("ts")
        dur_val = item.get("dur")
        if type(ts_val) not in (int, float) or type(dur_val) not in (int, float):
            continue
        start_us = float(str(ts_val))
        dur_us = float(str(dur_val))
        if dur_us <= 0.0:
            continue
        name = str(item.get("name", ""))
        cat = str(item.get("cat", ""))
        pid = int(str(item.get("pid", 0)))
        tid = int(str(item.get("tid", 0)))
        args_raw = item.get("args")
        args: Mapping[str, object] = args_raw if type(args_raw) is dict else {}
        rank_key = f"pid:{pid}"
        if _is_step_event(name, cat, "", args):
            steps_by_rank.setdefault(rank_key, []).append(
                _build_step_window(start_us, dur_us, args)
            )
        else:
            stream_key = f"{rank_key}:tid:{tid}"
            events_by_stream.setdefault(stream_key, []).append(
                _build_compute_event(
                    (rank_key, stream_key), start_us, dur_us, name, args
                )
            )
    return _ParsedTrace(
        kind="perfetto",
        file_label=path.name,
        steps_by_rank=steps_by_rank,
        events_by_stream=events_by_stream,
    )


def _extract_xevent_stats(
    ev: _XEventProto, stat_meta: Mapping[int, _XStatMetaProto]
) -> dict[str, object]:
    """Extract a dictionary of stat values from an XEvent proto."""
    out: dict[str, object] = {}
    for st in ev.stats:
        meta = stat_meta.get(st.metadata_id)
        if meta is None:
            continue
        key = str(meta.name)
        if st.str_value:
            out[key] = str(st.str_value)
        elif st.double_value != 0.0:
            out[key] = float(st.double_value)
        elif st.uint64_value != 0:
            out[key] = int(st.uint64_value)
        else:
            out[key] = int(st.int64_value)
    return out


def _ingest_xplane(
    plane: _XPlaneProto,
    rank_key: str,
    steps_by_rank: dict[str, list[_StepWindow]],
    events_by_stream: dict[str, list[_ComputeEvent]],
) -> None:
    """Ingest all lines from a device XPlane into step or compute streams."""
    ev_meta = dict(plane.event_metadata.items())
    stat_meta = dict(plane.stat_metadata.items())
    for line in plane.lines:
        line_name = str(line.name or line.display_name or "")
        line_id = int(line.id)
        line_base_us = float(line.timestamp_ns) / _NS_PER_US
        stream_key = f"{rank_key}:line:{line_id}:{line_name}"
        for ev in line.events:
            dur_us = float(ev.duration_ps) / _PS_PER_US
            if dur_us <= 0.0:
                continue
            start_us = line_base_us + float(ev.offset_ps) / _PS_PER_US
            meta = ev_meta.get(ev.metadata_id)
            ev_name = (
                str(meta.display_name or meta.name)
                if meta is not None
                else f"ev_{ev.metadata_id}"
            )
            args = _extract_xevent_stats(ev, stat_meta)
            if _is_step_event(ev_name, "", line_name, args):
                steps_by_rank.setdefault(rank_key, []).append(
                    _build_step_window(start_us, dur_us, args)
                )
            else:
                events_by_stream.setdefault(stream_key, []).append(
                    _build_compute_event(
                        (rank_key, stream_key), start_us, dur_us, ev_name, args
                    )
                )


def _discover_xplane_files(path: Path) -> list[Path]:
    """Discover `.xplane.pb` files from a single path or profile directory."""
    if path.is_file():
        return [path]
    sessions = merge_xplane.discover_sessions([str(path)])
    if len(sessions) > 1:
        dirs = ", ".join(str(s.directory) for s in sessions)
        msg = (
            f"Multiple XPlane sessions found under {path} ({dirs}); "
            "pass a single session directory"
        )
        raise errors.MegakernelConfigError(msg)
    if len(sessions) == 1:
        return [Path(cap.path) for cap in sessions[0].captures]
    return sorted(path.rglob("*.xplane.pb"))


def _parse_xplane_target(path: Path) -> _ParsedTrace:
    """Parse one or more `.xplane.pb` files using `vllm_torchtpu.tools.xplane_pb2`."""
    files = _discover_xplane_files(path)
    if not files:
        msg = f"No .xplane.pb files found under {path}"
        raise errors.MegakernelConfigError(msg)

    steps_by_rank: dict[str, list[_StepWindow]] = {}
    events_by_stream: dict[str, list[_ComputeEvent]] = {}
    for file_idx, pb_path in enumerate(files):
        space = vars(xplane_pb2)["XSpace"]()
        space.ParseFromString(pb_path.read_bytes())
        for plane in space.planes:
            pname = str(plane.name)
            if not pname.startswith("/device:") or "sparse" in pname.lower():
                continue
            rank_key = f"file:{file_idx}:plane:{plane.id}:{pname}"
            _ingest_xplane(plane, rank_key, steps_by_rank, events_by_stream)
    return _ParsedTrace(
        kind="xplane",
        file_label=path.name,
        steps_by_rank=steps_by_rank,
        events_by_stream=events_by_stream,
    )


def _steady_state_steps(steps: Sequence[_StepWindow]) -> list[_StepWindow]:
    """Sort step windows and drop warmup step 0 when at least 3 steps exist.

    Also drops step 0 when at least 2 steps exist if step 0 is a severe compilation
    outlier exceeding _WARMUP_OUTLIER_RATIO times the minimum duration of subsequent steps.
    """
    ordered = sorted(steps, key=lambda sw: sw.start_us)
    if len(ordered) >= 2:
        min_steady_step_dur_us = min(s.dur_us for s in ordered[1:])
        if ordered[0].dur_us > _WARMUP_OUTLIER_RATIO * min_steady_step_dur_us:
            return ordered[1:]
        if len(ordered) >= _MIN_STEPS_FOR_WARMUP_DROP:
            return ordered[1:]
    return ordered


def _scan_event_window(
    window_evs: Sequence[_ComputeEvent],
) -> tuple[list[float], list[_ComputeEvent]]:
    """Collect positive consecutive gaps and collectives within one event window."""
    gaps: list[float] = []
    collectives: list[_ComputeEvent] = []
    for idx, cur_ev in enumerate(window_evs):
        if cur_ev.collective_op is not None:
            collectives.append(cur_ev)
        if idx > 0:
            gap = cur_ev.start_us - window_evs[idx - 1].end_us
            if 0.0 < gap <= _MAX_INTRA_STEP_GAP_US:
                gaps.append(gap)
    return gaps, collectives


def _collect_stream_gaps_and_collectives(
    evs: Sequence[_ComputeEvent],
    steady_steps: Sequence[_StepWindow],
) -> tuple[list[float], list[_ComputeEvent]]:
    """Extract intra-step positive launch gaps and steady-state collective events."""
    ordered = sorted(evs, key=lambda e: e.start_us)
    if steady_steps:
        starts = [ev.start_us for ev in ordered]
        gaps: list[float] = []
        collectives: list[_ComputeEvent] = []
        for sw in steady_steps:
            lo = bisect_left(starts, sw.start_us)
            hi = bisect_left(starts, sw.end_us)
            w_gaps, w_colls = _scan_event_window(ordered[lo:hi])
            gaps.extend(w_gaps)
            collectives.extend(w_colls)
        return gaps, collectives

    trimmed = ordered[1:] if len(ordered) >= _MIN_EVENTS_FOR_WARMUP_DROP else ordered
    return _scan_event_window(trimmed)


def _summarize_collectives(
    collectives: Sequence[_ComputeEvent],
    default_group_size: int | None = None,
) -> list[CollectiveSummary]:
    """Group collective events by `(op, group_size, bytes)` and compute medians."""
    buckets: dict[tuple[str, int, int], list[float]] = {}
    for ev in collectives:
        grp = ev.group_size if ev.group_size is not None else default_group_size
        if ev.collective_op is None or grp is None or grp < _MIN_COLLECTIVE_GROUP_SIZE:
            continue
        key = (ev.collective_op, grp, ev.byte_size)
        buckets.setdefault(key, []).append(ev.dur_us)

    rows: list[CollectiveSummary] = []
    for (op_name, grp, nbytes), durs in sorted(buckets.items()):
        s_durs = sorted(durs)
        med_us = _percentile(s_durs, 0.5)
        hops = max(1, round(log2(max(2, grp))))
        rows.append(
            {
                "op": op_name,
                "group_size": grp,
                "bytes": nbytes,
                "latency_us": med_us,
                "hop_latency_us": round(med_us / hops, 4),
                "n": len(s_durs),
            }
        )
    return rows


def _summarize_step_coverage(
    steady_steps: Sequence[_StepWindow],
) -> StepCoverageSummary | None:
    """Compute `step_coverage` when step token or phase metadata is present."""
    total_active = 0
    total_elig = 0
    meta_count = 0
    for sw in steady_steps:
        if (
            sw.active_tokens is not None
            and sw.eligible_tokens is not None
            and sw.active_tokens > 0
        ):
            total_active += sw.active_tokens
            total_elig += sw.eligible_tokens
            meta_count += 1
        elif sw.phase is not None:
            total_active += 1
            total_elig += 1 if sw.phase in ("decode", "verify") else 0
            meta_count += 1
    if meta_count == 0 or total_active == 0:
        return None
    return {
        "fused_eligible_fraction": round(total_elig / total_active, 4),
        "basis": "trace",
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
    chip: str,
    model: str | None,
    phase: str,
    extra: Mapping[str, int | None],
) -> CalibrationSourceMeta:
    """Assemble provenance metadata for the calibrated output payload."""
    source_meta: CalibrationSourceMeta = {
        "kind": parsed.kind,
        "file": parsed.file_label,
        "chip": chip,
        "phase": phase,
    }
    if model is not None:
        source_meta["model"] = model
    batch_val = extra.get("batch")
    if batch_val is not None:
        source_meta["batch"] = batch_val
    tp_val = extra.get("tp")
    if tp_val is not None:
        source_meta["tp"] = tp_val
    return source_meta


def calibrate_trace(
    trace_path: str | Path,
    out_path: str | Path | None = None,
    *,
    chip: str = "tpu7x",
    model: str | None = None,
    phase: str = "decode",
    **extra_meta: int | None,
) -> CalibrationPayload:
    """Calibrate advisor priors from an XPlane or Perfetto trace file or directory."""
    parsed = _load_parsed_trace(trace_path)

    steady_by_rank = {
        rk: _steady_state_steps(steps) for rk, steps in parsed.steps_by_rank.items()
    }
    all_steady_steps = [
        sw for rk in sorted(steady_by_rank) for sw in steady_by_rank[rk]
    ]

    all_gaps: list[float] = []
    all_collectives: list[_ComputeEvent] = []
    for stream_key in sorted(parsed.events_by_stream):
        evs = parsed.events_by_stream[stream_key]
        if not evs:
            continue
        rk = evs[0].rank_key
        gaps, colls = _collect_stream_gaps_and_collectives(
            evs, steady_by_rank.get(rk, [])
        )
        all_gaps.extend(gaps)
        all_collectives.extend(colls)

    sorted_gaps = sorted(all_gaps)
    launch_gap: QuantileSummary = (
        {
            "p50": _percentile(sorted_gaps, 0.5),
            "p90": _percentile(sorted_gaps, 0.9),
            "n": len(sorted_gaps),
        }
        if sorted_gaps
        else {"n": 0}
    )

    payload: CalibrationPayload = {
        "schema": _MEASUREMENTS_SCHEMA,
        "source": _build_source_meta(parsed, chip, model, phase, extra_meta),
        "launch_gap_us": launch_gap,
        "collectives": _summarize_collectives(all_collectives, extra_meta.get("tp")),
    }

    cov = _summarize_step_coverage(all_steady_steps)
    if cov is not None:
        payload["step_coverage"] = cov

    if all_steady_steps:
        step_durs = sorted(sw.dur_us for sw in all_steady_steps)
        payload["step_time_us"] = {
            "p50": _percentile(step_durs, 0.5),
            "p90": _percentile(step_durs, 0.9),
            "n": len(step_durs),
        }

    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
