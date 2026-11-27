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
"""Unit tests for the megakernel advisor trace calibration engine."""

from __future__ import annotations

import dataclasses
import gzip
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from tests.megakernel.advisor_test_helpers import build_preset_workload
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import calibrate, constants, specs
from vllm_torchtpu.megakernel.core import errors
from vllm_torchtpu.tools import xplane_pb2 as _xplane_pb2

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

xplane_pb2: Any = _xplane_pb2

ALL_REDUCE_BYTES = 128 * 1024
ALL_TO_ALL_BYTES = 64 * 1024
TP_GROUP_SIZE = 8
WARMUP_STEP_DURATION_US = 140
WARMUP_OP_DURATION_US = 20
TAGGED_STAT_PAIR_LEN = 2
EXPECTED_PAIR_COUNT = 2
STEADY_STEP_SPECS: tuple[tuple[int, int, int, str, int, int], ...] = (
    (1, 200, 98, "decode", 16, 16),
    (2, 400, 100, "verify", 16, 16),
    (3, 600, 100, "decode", 16, 16),
    (4, 800, 102, "prefill", 16, 0),
)
STEADY_OP_SPECS: tuple[tuple[int, str, float, int, int], ...] = (
    (2, "qkv_proj", 5.0, 10, 0),
    (3, "all-reduce", 17.0, 12, ALL_REDUCE_BYTES),
    (4, "attn_core", 31.5, 15, 0),
    (5, "all-to-all", 50.0, 18, ALL_TO_ALL_BYTES),
)

XLA_MODULE_NUM_STEPS = 4
XLA_MODULE_DEFAULT_LAYERS = 60
XLA_MODULE_START_OFFSET_US = 1000.0
XLA_MODULE_FIRST_OP_OFFSET_US = 5800.0
XLA_MODULE_POST_TAIL_US = 10.0
XLA_MODULE_AUX_DUR_US = 200.0
XLA_MODULE_AUX_OP_OFFSET_US = 30.0
XLA_MODULE_AUX_OP_DUR_US = 150.0
XLA_MODULE_STRIDE_US = 30_000.0
XLA_MODULE_AUX_OFFSET_US = 25_000.0


@dataclass(frozen=True, slots=True)
class _XlaModuleSpec:
    """Configuration for synthetic multi-step XLA Modules traces."""

    num_layers: int = XLA_MODULE_DEFAULT_LAYERS
    ranks: tuple[int | str, ...] = (1,)
    host_wraps: bool = False
    overhanging_tail_us: float = 0.0
    idle_scale: float = 1.0
    num_ops_override: int | None = None
    module_name: str = "jit_main_decode_step"
    module_step_num: bool = False


@dataclass(frozen=True, slots=True)
class _XPlaneExtras:
    """Optional `XEventMetadata.stats` and aggregated occurrence events."""

    metadata_stats: Mapping[str, Mapping[str, object]] | None = None
    num_occurrences_events: Sequence[tuple[str, str, int, float]] = ()


def _write_json_trace(tmp_path: Path, filename: str, payload: object) -> Path:
    """Write a JSON payload to `tmp_path / filename` and return its path."""
    trace_file = tmp_path / filename
    trace_file.write_text(json.dumps(payload), encoding="utf-8")
    return trace_file


def _write_xplane_dir(root: Path, captures: Mapping[str, bytes]) -> Path:
    """Create a profile capture directory populated with `.xplane.pb` bytes."""
    session_dir = root / "plugins" / "profile" / "2026_11_04"
    session_dir.mkdir(parents=True, exist_ok=True)
    for filename, raw_bytes in captures.items():
        (session_dir / filename).write_bytes(raw_bytes)
    return root


def _xla_module_step_geometry(
    num_layers: int = XLA_MODULE_DEFAULT_LAYERS,
    *,
    num_ops_override: int | None = None,
    idle_scale: float = 1.0,
) -> tuple[int, float, float, float, float, float]:
    """Return `(num_ops, op_dur_us, gap_us, raw_mod_dur_us, span_us, idle_us)`."""
    num_ops = (
        num_ops_override
        if num_ops_override is not None
        else constants.SUBLAYER_BOUNDARIES_PER_LAYER * num_layers
    )
    base_idle_us = 240.0 + 1.5 * num_layers
    idle_us = base_idle_us * idle_scale
    active_span_us = 6800.0 + 10.0 * num_layers + (idle_scale - 1.0) * base_idle_us
    busy_us = active_span_us - idle_us
    op_duration_us = busy_us / num_ops
    gap_us = idle_us / (num_ops - 1)
    raw_module_duration_us = (
        (XLA_MODULE_FIRST_OP_OFFSET_US - XLA_MODULE_START_OFFSET_US)
        + active_span_us
        + XLA_MODULE_POST_TAIL_US
    )
    return (
        num_ops,
        op_duration_us,
        gap_us,
        raw_module_duration_us,
        active_span_us,
        idle_us,
    )


def _synthetic_perfetto_events() -> list[dict[str, object]]:
    """Return standard warmup and steady-state Perfetto trace events."""
    events: list[dict[str, object]] = [
        {
            "ph": "X",
            "cat": "step",
            "name": "model_step",
            "pid": 1,
            "tid": 10,
            "ts": 0.0,
            "dur": float(WARMUP_STEP_DURATION_US),
            "args": {
                "step": 0,
                "phase": "decode",
                "active_tokens": 16,
                "fused_eligible_tokens": 16,
            },
        },
        {
            "ph": "X",
            "cat": "xla_op",
            "name": "linear_proj",
            "pid": 1,
            "tid": 1,
            "ts": 10.0,
            "dur": float(WARMUP_OP_DURATION_US),
            "args": {},
        },
        {
            "ph": "X",
            "cat": "xla_op",
            "name": "all-reduce",
            "pid": 1,
            "tid": 1,
            "ts": 55.0,
            "dur": float(WARMUP_OP_DURATION_US),
            "args": {"group_size": TP_GROUP_SIZE, "bytes": ALL_REDUCE_BYTES},
        },
    ]
    for (
        step_num,
        base_us,
        duration_us,
        phase_str,
        active_tokens,
        eligible_tokens,
    ) in STEADY_STEP_SPECS:
        events.append(
            {
                "ph": "X",
                "cat": "step",
                "name": "model_step",
                "pid": 1,
                "tid": 10,
                "ts": float(base_us),
                "dur": float(duration_us),
                "args": {
                    "step": step_num,
                    "phase": phase_str,
                    "active_tokens": active_tokens,
                    "fused_eligible_tokens": eligible_tokens,
                },
            }
        )
        for _, op_name, rel_us, op_duration_us, collective_bytes in STEADY_OP_SPECS:
            args = (
                {"group_size": TP_GROUP_SIZE, "bytes": collective_bytes}
                if collective_bytes > 0
                else {}
            )
            events.append(
                {
                    "ph": "X",
                    "cat": "xla_op",
                    "name": op_name,
                    "pid": 1,
                    "tid": 1,
                    "ts": float(base_us) + rel_us,
                    "dur": float(op_duration_us),
                    "args": args,
                }
            )
    return events


def _xla_module_perfetto_events(spec: _XlaModuleSpec) -> list[dict[str, object]]:
    """Return multi-step Perfetto events for an `_XlaModuleSpec` fixture."""
    num_ops, op_duration_us, gap_us, raw_module_duration_us, _, _ = (
        _xla_module_step_geometry(
            spec.num_layers,
            num_ops_override=spec.num_ops_override,
            idle_scale=spec.idle_scale,
        )
    )
    events: list[dict[str, object]] = []
    wrap_pids: tuple[int | str, ...] = (0,) if spec.host_wraps else spec.ranks
    for step_idx in range(XLA_MODULE_NUM_STEPS):
        base_us = step_idx * XLA_MODULE_STRIDE_US
        for wrap_pid in wrap_pids:
            main_args = {"step_num": step_idx} if spec.module_step_num else {}
            for module_label, offset_us, dur_us, module_args in (
                (
                    spec.module_name,
                    XLA_MODULE_START_OFFSET_US,
                    raw_module_duration_us,
                    main_args,
                ),
                (
                    "jit_aux_sampler_stub",
                    XLA_MODULE_AUX_OFFSET_US,
                    XLA_MODULE_AUX_DUR_US,
                    {},
                ),
            ):
                events.append(
                    {
                        "ph": "X",
                        "cat": "XLA Modules",
                        "name": module_label,
                        "pid": wrap_pid,
                        "tid": 1,
                        "ts": base_us + offset_us,
                        "dur": dur_us,
                        "args": module_args,
                    }
                )
        for rank_pid in spec.ranks:
            for op_idx in range(num_ops):
                extra_tail = (
                    spec.overhanging_tail_us
                    if op_idx == num_ops - 1 and step_idx > 0
                    else 0.0
                )
                events.append(
                    {
                        "ph": "X",
                        "cat": "xla_op",
                        "name": f"fused_layer_op_{op_idx}",
                        "pid": rank_pid,
                        "tid": 2,
                        "ts": (
                            base_us
                            + XLA_MODULE_FIRST_OP_OFFSET_US
                            + op_idx * (op_duration_us + gap_us)
                        ),
                        "dur": op_duration_us + extra_tail,
                    }
                )
            events.append(
                {
                    "ph": "X",
                    "cat": "xla_op",
                    "name": "aux_sampler_op",
                    "pid": rank_pid,
                    "tid": 2,
                    "ts": (
                        base_us + XLA_MODULE_AUX_OFFSET_US + XLA_MODULE_AUX_OP_OFFSET_US
                    ),
                    "dur": XLA_MODULE_AUX_OP_DUR_US,
                }
            )
    return events


def _build_perfetto_trace(
    *,
    events: Sequence[Mapping[str, object]] = (),
    steps: Sequence[tuple[int, str, int, int]] = (),
    synthetic_default: bool = False,
    xla_module: _XlaModuleSpec | None = None,
) -> dict[str, object]:
    """Build a synthetic Perfetto `traceEvents` dictionary for calibration tests."""
    built: list[dict[str, object]] = [dict(event) for event in events]
    for step_num, phase, timestamp_us, duration_us in steps:
        built.append(
            {
                "ph": "X",
                "name": f"Step {step_num}",
                "cat": "ModelStep",
                "ts": timestamp_us,
                "dur": duration_us,
                "pid": 1,
                "tid": 1,
                "args": {"step_num": step_num, "phase": phase},
            }
        )
    if synthetic_default:
        built.extend(_synthetic_perfetto_events())
    if xla_module is not None:
        built.extend(_xla_module_perfetto_events(xla_module))
    return {"traceEvents": built}


def _xplane_stat_entry(value: object) -> tuple[str, object]:
    """Map a scalar or tagged stat tuple to `(protobuf_field_name, value)`."""
    if isinstance(value, tuple) and len(value) == TAGGED_STAT_PAIR_LEN:
        kind, raw_val = value
        field_by_kind = {
            "ref": "ref_value",
            "bytes": "bytes_value",
            "uint64": "uint64_value",
            "double": "double_value",
        }
        return field_by_kind[str(kind)], raw_val
    if type(value) is int:
        return "int64_value", value
    if type(value) is float:
        return "double_value", value
    if isinstance(value, bytes):
        return "bytes_value", value
    return "str_value", str(value)


def _synthetic_xplane_lines(
    time_shift_us: int,
) -> list[tuple[str, int, list[tuple[str, float, float, Mapping[str, object]]]]]:
    """Return synthetic `Steps` and `XLA Ops` line definitions for XPlane."""
    step_events: list[tuple[str, float, float, Mapping[str, object]]] = [
        (
            "Step",
            float(time_shift_us),
            float(WARMUP_STEP_DURATION_US),
            {"step_num": 0, "phase": "decode"},
        )
    ]
    op_events: list[tuple[str, float, float, Mapping[str, object]]] = [
        ("qkv_proj", float(time_shift_us + 10), float(WARMUP_OP_DURATION_US), {}),
        ("all-reduce.1", float(time_shift_us + 55), float(WARMUP_OP_DURATION_US), {}),
    ]
    for (
        step_num,
        base_us,
        duration_us,
        phase_str,
        active_tokens,
        eligible_tokens,
    ) in STEADY_STEP_SPECS:
        step_start_us = float(time_shift_us + base_us)
        step_events.append(
            (
                "Step",
                step_start_us,
                float(duration_us),
                {
                    "step_num": step_num,
                    "phase": phase_str,
                    "active_tokens": active_tokens,
                    "fused_eligible_tokens": eligible_tokens,
                },
            )
        )
        for _, op_name, rel_us, op_duration_us, collective_bytes in STEADY_OP_SPECS:
            op_stats: dict[str, object] = (
                {"group_size": TP_GROUP_SIZE, "bytes": collective_bytes}
                if collective_bytes > 0
                else {}
            )
            xplane_op = (
                "all-reduce.1"
                if op_name == "all-reduce"
                else ("all-to-all.2" if op_name == "all-to-all" else op_name)
            )
            op_events.append(
                (xplane_op, step_start_us + rel_us, float(op_duration_us), op_stats)
            )
    return [("Steps", 0, step_events), ("XLA Ops", 0, op_events)]


def _xla_module_xplane_lines(
    spec: _XlaModuleSpec,
) -> list[tuple[str, int, list[tuple[str, float, float, Mapping[str, object]]]]]:
    """Return `XLA Modules` and `XLA Ops` line definitions for `_XlaModuleSpec`."""
    num_ops, op_duration_us, gap_us, raw_module_duration_us, _, _ = (
        _xla_module_step_geometry(
            spec.num_layers,
            num_ops_override=spec.num_ops_override,
            idle_scale=spec.idle_scale,
        )
    )
    mod_events: list[tuple[str, float, float, Mapping[str, object]]] = []
    op_events: list[tuple[str, float, float, Mapping[str, object]]] = []
    for step_idx in range(XLA_MODULE_NUM_STEPS):
        base_us = step_idx * XLA_MODULE_STRIDE_US
        mod_stats: dict[str, object] = (
            {"step_num": step_idx} if spec.module_step_num else {}
        )
        mod_events.append(
            (
                spec.module_name,
                base_us + XLA_MODULE_START_OFFSET_US,
                raw_module_duration_us,
                mod_stats,
            )
        )
        for op_idx in range(num_ops):
            op_start_us = (
                base_us
                + XLA_MODULE_FIRST_OP_OFFSET_US
                + op_idx * (op_duration_us + gap_us)
            )
            op_events.append(("fused_layer_op", op_start_us, op_duration_us, {}))
        mod_events.append(
            (
                "jit_aux_sampler_stub",
                base_us + XLA_MODULE_AUX_OFFSET_US,
                XLA_MODULE_AUX_DUR_US,
                {},
            )
        )
        op_events.append(
            (
                "fused_layer_op",
                base_us + XLA_MODULE_AUX_OFFSET_US + XLA_MODULE_AUX_OP_OFFSET_US,
                XLA_MODULE_AUX_OP_DUR_US,
                {},
            )
        )
    return [("XLA Modules", 0, mod_events), ("XLA Ops", 0, op_events)]


def _collect_xplane_lines(
    *,
    time_shift_us: int,
    synthetic_default: bool,
    xla_module: _XlaModuleSpec | None,
    lines: Sequence[
        tuple[str, int, Sequence[tuple[str, float, float, Mapping[str, object]]]]
    ],
) -> list[tuple[str, int, Sequence[tuple[str, float, float, Mapping[str, object]]]]]:
    """Assemble explicit, synthetic, and XLA Module line tuples for XPlane."""
    collected: list[
        tuple[str, int, Sequence[tuple[str, float, float, Mapping[str, object]]]]
    ] = list(lines)
    if synthetic_default:
        collected.extend(_synthetic_xplane_lines(time_shift_us))
    if xla_module is not None:
        collected.extend(_xla_module_xplane_lines(xla_module))
    return collected


def _intern_xplane_name(table: dict[str, int], meta_map: object, label: str) -> int:
    """Intern `label` into `table` and protobuf metadata map."""
    if label not in table:
        new_id = len(table) + 1
        table[label] = new_id
        meta_table: Any = meta_map
        entry = meta_table[new_id]
        entry.id, entry.name = new_id, label
    return table[label]


def _build_xplane_bytes(
    *,
    time_shift_us: int = 0,
    synthetic_default: bool = False,
    xla_module: _XlaModuleSpec | None = None,
    lines: Sequence[
        tuple[str, int, Sequence[tuple[str, float, float, Mapping[str, object]]]]
    ] = (),
    extras: _XPlaneExtras | None = None,
) -> bytes:
    """Serialize a synthetic `XSpace` proto from step, module, or custom lines."""
    ps_per_us = int(constants.PS_PER_US)
    space = xplane_pb2.XSpace()
    space.hostnames.append("tpu-worker-0")
    host_specs = (
        ((99, "/host:metadata"), (98, "/host:CPU")) if synthetic_default else ()
    )
    for plane_id, plane_name in host_specs:
        host_plane = space.planes.add()
        host_plane.id, host_plane.name = plane_id, plane_name

    dev_plane = space.planes.add()
    dev_plane.id, dev_plane.name = 1, "/device:TPU:0"
    event_ids: dict[str, int] = {}
    stat_ids: dict[str, int] = {}

    xplane_extras = extras or _XPlaneExtras()
    pending_stats: list[tuple[object, Mapping[str, object]]] = []
    for event_name, event_stats in (xplane_extras.metadata_stats or {}).items():
        eid = _intern_xplane_name(event_ids, dev_plane.event_metadata, event_name)
        pending_stats.append((dev_plane.event_metadata[eid], event_stats))

    all_lines = _collect_xplane_lines(
        time_shift_us=time_shift_us,
        synthetic_default=synthetic_default,
        xla_module=xla_module,
        lines=lines,
    )
    xline_by_name: dict[str, object] = {}
    for line_idx, (line_name, line_ts_ns, event_specs) in enumerate(all_lines, start=1):
        xline = dev_plane.lines.add()
        xline.id, xline.name, xline.timestamp_ns = line_idx, line_name, line_ts_ns
        xline_by_name[line_name] = xline
        for event_name, start_us, duration_us, stats_map in event_specs:
            event = xline.events.add()
            event.metadata_id = _intern_xplane_name(
                event_ids, dev_plane.event_metadata, event_name
            )
            event.offset_ps = round(start_us * ps_per_us)
            event.duration_ps = round(duration_us * ps_per_us)
            if stats_map:
                pending_stats.append((event, stats_map))

    for holder, stats_map in pending_stats:
        stats_owner: Any = holder
        for key, raw_val in stats_map.items():
            field_name, value = _xplane_stat_entry(raw_val)
            stat = stats_owner.stats.add()
            stat.metadata_id = _intern_xplane_name(
                stat_ids, dev_plane.stat_metadata, key
            )
            setattr(
                stat,
                field_name,
                _intern_xplane_name(stat_ids, dev_plane.stat_metadata, str(value))
                if field_name == "ref_value"
                else value,
            )

    for (
        line_name,
        event_name,
        occurrences,
        duration_us,
    ) in xplane_extras.num_occurrences_events:
        agg_line: Any = xline_by_name[line_name]
        agg_event = agg_line.events.add()
        agg_event.metadata_id = _intern_xplane_name(
            event_ids, dev_plane.event_metadata, event_name
        )
        agg_event.num_occurrences = occurrences
        agg_event.duration_ps = round(duration_us * ps_per_us)

    return bytes(space.SerializeToString())


def _advise_qwen35_397b_decode(
    measurements_source: Path,
) -> advisor.AdvisorReport:
    """Run `advisor.advise` for Qwen3.5-397B decode on `tpu7x` with calibration."""
    profile = advisor.resolve_model_profile("Qwen/Qwen3.5-397B-A17B-FP8")
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    return advisor.advise(
        profile,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        chip_spec,
        measurements=measurements_source,
    )


def test_calibrate_perfetto_extracts_launch_gap_and_collectives(
    tmp_path: Path,
) -> None:
    """Verify Perfetto CLI calibration extracts steady-state gaps and collectives."""
    trace_gz = tmp_path / "synthetic_perfetto.json.gz"
    trace_gz.write_bytes(
        gzip.compress(
            json.dumps(_build_perfetto_trace(synthetic_default=True)).encode("utf-8")
        )
    )
    out_json = tmp_path / "perfetto_measurements.json"

    requested_batch = 16
    requested_tp = 8
    exit_code = advisor.main(
        [
            "calibrate",
            "--trace",
            str(trace_gz),
            "--chip",
            "tpu7x",
            "--model",
            "Qwen/Qwen3.5-397B-A17B-FP8",
            "--batch",
            str(requested_batch),
            "--tp",
            str(requested_tp),
            "--out",
            str(out_json),
        ]
    )
    assert exit_code == 0
    assert out_json.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == constants.MEASUREMENTS_SCHEMA
    assert payload["source"]["kind"] == "perfetto"
    assert payload["source"]["batch"] == requested_batch
    assert payload["source"]["tp"] == requested_tp

    expected_gaps_count = 3 * 3
    expected_steady_steps = 3
    launch_gap = payload["launch_gap_us"]
    assert launch_gap["p50"] == pytest.approx(2.5)
    assert launch_gap["p90"] == pytest.approx(3.5)
    assert launch_gap["n"] == expected_gaps_count

    collectives_by_op = {item["op"]: item for item in payload["collectives"]}
    assert set(collectives_by_op) == {"all-reduce", "all-to-all"}
    assert collectives_by_op["all-reduce"]["group_size"] == TP_GROUP_SIZE
    assert collectives_by_op["all-reduce"]["bytes"] == ALL_REDUCE_BYTES
    assert collectives_by_op["all-reduce"]["latency_us"] == pytest.approx(12.0)
    assert collectives_by_op["all-reduce"]["mean_latency_us"] == pytest.approx(12.0)
    assert collectives_by_op["all-reduce"]["n"] == expected_steady_steps
    assert collectives_by_op["all-to-all"]["bytes"] == ALL_TO_ALL_BYTES
    assert collectives_by_op["all-to-all"]["latency_us"] == pytest.approx(18.0)
    assert collectives_by_op["all-to-all"]["n"] == expected_steady_steps

    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)
    assert payload["step_coverage"]["basis"] == constants.STEP_COVERAGE_BASIS
    assert payload["step_time_us"]["p50"] == pytest.approx(100.0)
    assert payload["step_time_us"]["n"] == expected_steady_steps

    report = _advise_qwen35_397b_decode(out_json)
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["launch_gap_us"]["value"] == pytest.approx(2.5)
    assert report.metrics["collective_call_us_xla"]["tag"] == "MEASURED"
    assert report.metrics["collective_call_us_xla"]["value"] == pytest.approx(12.0)
    assert report.metrics["step_coverage"]["tag"] == "MEASURED"
    assert report.metrics["step_coverage"]["value"] == pytest.approx(0.75)


def test_calibrate_xplane_roundtrip_with_in_repo_parser(tmp_path: Path) -> None:
    """Verify single-file and multi-rank XPlane calibration via `xplane_pb2`."""
    single_pb = tmp_path / "single_rank.xplane.pb"
    single_pb.write_bytes(_build_xplane_bytes(synthetic_default=True, time_shift_us=0))

    out_single = tmp_path / "xplane_single_measurements.json"
    options = advisor.CalibrateOptions(chip="tpu7x", model="Qwen/Qwen3.5-397B-A17B-FP8")
    payload = advisor.calibrate_trace(single_pb, out_single, options=options)
    expected_single_gaps = 3 * 3
    expected_multi_gaps = 2 * expected_single_gaps
    expected_multi_coll_n = 6
    assert out_single.is_file()
    assert payload["schema"] == constants.MEASUREMENTS_SCHEMA
    assert payload["source"]["kind"] == "xplane"
    assert payload["launch_gap_us"]["p50"] == pytest.approx(2.5)
    assert payload["launch_gap_us"]["p90"] == pytest.approx(3.5)
    assert payload["launch_gap_us"]["n"] == expected_single_gaps
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)
    assert payload["step_time_us"]["p50"] == pytest.approx(100.0)

    decode_root = _write_xplane_dir(
        tmp_path / "decode_trace",
        {
            "rank0_worker0.xplane.pb": _build_xplane_bytes(
                synthetic_default=True, time_shift_us=0
            ),
            "rank1_worker1.xplane.pb": _build_xplane_bytes(
                synthetic_default=True, time_shift_us=7
            ),
        },
    )
    out_multi = tmp_path / "xplane_multi_measurements.json"
    exit_code = advisor.main(
        ["calibrate", "--trace", str(decode_root), "--out", str(out_multi)]
    )
    assert exit_code == 0
    multi_payload = json.loads(out_multi.read_text(encoding="utf-8"))
    assert multi_payload["launch_gap_us"]["p50"] == pytest.approx(2.5)
    assert multi_payload["launch_gap_us"]["p90"] == pytest.approx(3.5)
    assert multi_payload["launch_gap_us"]["n"] == expected_multi_gaps
    multi_coll = {item["op"]: item for item in multi_payload["collectives"]}
    assert multi_coll["all-reduce"]["n"] == expected_multi_coll_n
    assert multi_coll["all-reduce"]["latency_us"] == pytest.approx(12.0)
    assert multi_coll["all-to-all"]["n"] == expected_multi_coll_n
    assert multi_coll["all-to-all"]["latency_us"] == pytest.approx(18.0)

    report = _advise_qwen35_397b_decode(out_multi)
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["collective_call_us_xla"]["tag"] == "MEASURED"
    assert report.metrics["collective_call_us_xla"]["value"] == pytest.approx(12.0)


def test_calibrate_float_stats_and_empty_gaps_preserve_assumed_prior(
    tmp_path: Path,
) -> None:
    """Verify float token stats parse cleanly and 0-gap traces keep ASSUMED."""
    trace_file = _write_json_trace(
        tmp_path,
        "single_op_float_stats.json",
        _build_perfetto_trace(
            events=[
                {
                    "ph": "X",
                    "name": "Step",
                    "cat": "step",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": 50.0,
                    "args": {
                        "step_num": 1.0,
                        "phase": "decode",
                        "active_tokens": 16.0,
                        "fused_eligible_tokens": 12.0,
                    },
                },
                {
                    "ph": "X",
                    "name": "all-reduce.1",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": 110.0,
                    "dur": 9.0,
                    "args": {"group_size": 8.0, "bytes": float(ALL_REDUCE_BYTES)},
                },
            ]
        ),
    )
    out_json = tmp_path / "empty_gap_measurements.json"
    payload = advisor.calibrate_trace(trace_file, out_json)
    assert "launch_gap_us" not in payload
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)
    assert advisor.load_measurements(out_json).launch_gap_n == 0

    for model_slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(model_slug)
        model_ar_bytes = round(
            profile.hidden_size * 16 * constants.DTYPE_BYTES[profile.act_dtype]
        )
        model_trace = _write_json_trace(
            tmp_path,
            f"empty_gap_{model_slug}.json",
            [
                {
                    "name": "Step 1",
                    "cat": "step",
                    "ph": "X",
                    "pid": 1,
                    "tid": 0,
                    "ts": 100.0,
                    "dur": 60.0,
                    "args": {
                        "phase": "decode",
                        "active_tokens": 16.0,
                        "fused_eligible_tokens": 12.0,
                    },
                },
                {
                    "name": "all-reduce.1",
                    "cat": "op",
                    "ph": "X",
                    "pid": 1,
                    "tid": 2,
                    "ts": 110.0,
                    "dur": 9.0,
                    "args": {"group_size": 8.0, "bytes": float(model_ar_bytes)},
                },
            ],
        )
        model_out = tmp_path / f"empty_gap_{model_slug}.json.out"
        advisor.calibrate_trace(model_trace, model_out)
        report = advisor.advise(
            profile,
            advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
            advisor.resolve_chip_spec("tpu7x"),
            measurements=model_out,
        )
        assert report.metrics["launch_gap_us"]["tag"] == "ASSUMED"
        assert report.metrics["collective_call_us_xla"]["tag"] == "MEASURED"
        assert report.metrics["step_coverage"]["tag"] == "MEASURED"


def test_calibrate_rejects_invalid_inputs_and_handles_missing_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify input validation, collective group fallback, and missing tool errors."""
    phased_root = tmp_path / "phased_run"
    raw_bytes = _build_xplane_bytes(synthetic_default=True, time_shift_us=0)
    for phase_dir in ("prefill_only", "decode_only"):
        _write_xplane_dir(
            phased_root / phase_dir, {"rank0_worker0.xplane.pb": raw_bytes}
        )
    with pytest.raises(
        errors.MegakernelConfigError, match="Multiple XPlane sessions found"
    ):
        advisor.calibrate_trace(phased_root)

    with pytest.raises(errors.MegakernelConfigError, match="Missing trace path"):
        advisor.calibrate_trace(tmp_path / "nonexistent.json")

    bad_ext = tmp_path / "trace.txt"
    bad_ext.write_text("plain text", encoding="utf-8")
    with pytest.raises(errors.MegakernelConfigError, match="Unsupported trace format"):
        advisor.calibrate_trace(bad_ext)

    empty_dir = tmp_path / "empty_xplane_dir"
    empty_dir.mkdir()
    with pytest.raises(errors.MegakernelConfigError, match=r"No \.xplane\.pb files"):
        advisor.calibrate_trace(empty_dir)

    ungrouped_trace = _write_json_trace(
        tmp_path,
        "unannotated_collective.json",
        _build_perfetto_trace(
            events=[
                {
                    "ph": "X",
                    "name": "all-reduce-start.1",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": 100.0,
                    "dur": 9.0,
                    "args": {"bytes": 16384},
                },
                {
                    "ph": "X",
                    "name": "all-reduce-done.1",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": 112.0,
                    "dur": 1.0,
                    "args": {"group_size": 4, "bytes": 16384},
                },
            ]
        ),
    )
    assert advisor.calibrate_trace(ungrouped_trace)["collectives"] == []

    fallback_tp = 4
    tp_payload = advisor.calibrate_trace(
        ungrouped_trace, options=advisor.CalibrateOptions(tp=fallback_tp)
    )
    assert len(tp_payload["collectives"]) == 1
    assert tp_payload["collectives"][0]["group_size"] == fallback_tp

    monkeypatch.setattr(calibrate, "merge_xplane", None)
    with pytest.raises(advisor.XPlaneToolUnavailableError):
        advisor.calibrate_trace(empty_dir)

    monkeypatch.setattr(calibrate, "xplane_pb2", None)
    single_pb = tmp_path / "single.xplane.pb"
    single_pb.write_bytes(raw_bytes)
    with pytest.raises(errors.MegakernelConfigError, match="xplane_pb2"):
        advisor.calibrate_trace(single_pb)


@pytest.mark.parametrize(
    ("steps", "expected_n", "expected_p50_us"),
    [
        ([(0, "decode", 0, 30_000), (1, "decode", 30_000, 5_000)], 1, 5_000.0),
        ([(0, "decode", 0, 10_000), (1, "decode", 10_000, 5_000)], 2, 7_500.0),
        ([(0, "decode", 0, 5_200), (1, "decode", 5_200, 5_000)], 2, 5_100.0),
        (
            [
                (0, "decode", 0, 5_100),
                (1, "decode", 6_000, 5_000),
                (2, "decode", 12_000, 5_050),
            ],
            2,
            5_025.0,
        ),
    ],
)
def test_steady_state_steps_warmup_drop_policy(
    tmp_path: Path,
    steps: list[tuple[int, str, int, int]],
    expected_n: int,
    expected_p50_us: float,
) -> None:
    """Drop warmup step 0 strictly above 2.0x or >=3 steps; keep 2.0x boundary."""
    trace_file = _write_json_trace(
        tmp_path, "warmup_policy.json", _build_perfetto_trace(steps=steps)
    )
    calibrated_result = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="decode")
    )
    assert calibrated_result["step_time_us"]["n"] == expected_n
    assert calibrated_result["step_time_us"]["p50"] == pytest.approx(expected_p50_us)


def test_calibrate_trace_filters_by_requested_phase(tmp_path: Path) -> None:
    """Verify `calibrate_trace` isolates step metrics by requested phase."""
    trace_file = _write_json_trace(
        tmp_path,
        "filter_trace.json",
        _build_perfetto_trace(
            steps=[
                (0, "decode", 1000, 2000),
                (1, "decode", 5000, 2000),
                (2, "decode", 8000, 2100),
                (3, "prefill", 12000, 8000),
            ]
        ),
    )

    expected_decode = 2
    expected_mixed = 3
    calibrated_decode = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="decode")
    )
    assert calibrated_decode["step_time_us"]["n"] == expected_decode
    assert calibrated_decode["step_time_us"]["p50"] == pytest.approx(2050.0)
    assert calibrated_decode["source"]["phase"] == "decode"

    calibrated_prefill = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="prefill")
    )
    assert calibrated_prefill["step_time_us"]["n"] == 1
    assert calibrated_prefill["step_time_us"]["p50"] == pytest.approx(8000.0)
    assert calibrated_prefill["source"]["phase"] == "prefill"

    calibrated_mixed = calibrate.calibrate_trace(
        trace_file, options=calibrate.CalibrateOptions(phase="mixed")
    )
    assert calibrated_mixed["step_time_us"]["n"] == expected_mixed
    assert calibrated_mixed["source"]["phase"] == "mixed"

    with pytest.raises(errors.MegakernelConfigError, match="found phases:"):
        calibrate.calibrate_trace(
            trace_file, options=calibrate.CalibrateOptions(phase="verify")
        )


def test_calibrate_wraps_corrupt_json_and_xplane_errors(tmp_path: Path) -> None:
    """Verify malformed Perfetto JSON/gzip and corrupt XPlane bytes wrap cleanly."""
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(
        errors.MegakernelConfigError, match="Failed to read Perfetto trace"
    ):
        advisor.calibrate_trace(bad_json)

    bad_gz = tmp_path / "corrupt.json.gz"
    bad_gz.write_bytes(b"not a gzip payload")
    with pytest.raises(
        errors.MegakernelConfigError, match="Failed to read Perfetto trace"
    ):
        advisor.calibrate_trace(bad_gz)

    bad_pb = tmp_path / "corrupt.xplane.pb"
    bad_pb.write_bytes(b"\x08\x80\x80\x80")
    with pytest.raises(
        errors.MegakernelConfigError, match="Failed to parse XPlane trace"
    ):
        advisor.calibrate_trace(bad_pb)


def test_calibrate_ignores_bool_scalars_in_json_events(tmp_path: Path) -> None:
    """Verify JSON booleans are not coerced to 0/1 ints in collective/token stats."""
    bool_trace = _write_json_trace(
        tmp_path,
        "bool_scalars.json",
        _build_perfetto_trace(
            events=[
                {
                    "ph": "X",
                    "name": "Step",
                    "cat": "step",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": 50.0,
                    "args": {
                        "phase": "decode",
                        "active_tokens": True,
                        "fused_eligible_tokens": False,
                    },
                },
                {
                    "ph": "X",
                    "name": "all-reduce.1",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": 110.0,
                    "dur": 8.0,
                    "args": {"group_size": True, "bytes": True},
                },
            ]
        ),
    )
    payload = advisor.calibrate_trace(bool_trace)
    assert payload["collectives"] == []
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(1.0)


def test_calibrate_cli_validates_chip_tp_and_batch_upfront(tmp_path: Path) -> None:
    """Verify `advisor calibrate` and `CalibrateOptions` fail closed upfront."""
    empty_trace = _write_json_trace(tmp_path, "empty.json", {"traceEvents": []})
    out_json = tmp_path / "out.json"
    base_argv = ["calibrate", "--trace", str(empty_trace), "--out", str(out_json)]

    with pytest.raises(errors.MegakernelConfigError, match="Unknown chip"):
        advisor.main([*base_argv, "--chip", "invalid_chip_xyz"])
    with pytest.raises(errors.MegakernelConfigError):
        advisor.main([*base_argv, "--model", "unknown_model_xyz"])
    with pytest.raises(errors.MegakernelConfigError, match="power of 2 >= 1"):
        advisor.main([*base_argv, "--tp", "-4"])
    with pytest.raises(errors.MegakernelConfigError, match="power of 2 >= 1"):
        advisor.main([*base_argv, "--tp", "3"])
    with pytest.raises(errors.MegakernelConfigError, match="batch must be >= 1"):
        advisor.main([*base_argv, "--batch", "0"])
    with pytest.raises(errors.MegakernelConfigError, match="--num-layers must be >= 1"):
        advisor.main([*base_argv, "--num-layers", "0"])

    for bad_phase, bad_batch, bad_tp, bad_layers in (
        ("bogus_phase", None, None, None),
        ("decode", 0, None, None),
        ("decode", -5, None, None),
        ("decode", None, 3, None),
        ("decode", None, None, 0),
    ):
        with pytest.raises(errors.MegakernelConfigError):
            advisor.CalibrateOptions(
                phase=bad_phase,
                batch=bad_batch,
                tp=bad_tp,
                num_layers=bad_layers,
            )
    with pytest.raises(errors.MegakernelConfigError, match="Unknown chip"):
        advisor.calibrate_trace(
            empty_trace, options=advisor.CalibrateOptions(chip="invalid_chip_xyz")
        )
    with pytest.raises(errors.MegakernelConfigError):
        advisor.calibrate_trace(
            empty_trace, options=advisor.CalibrateOptions(model="unknown_model_xyz")
        )
    help_text = " ".join(advisor.build_calibrate_arg_parser().format_help().split())
    assert "pp > 1" in help_text


@pytest.mark.parametrize("trace_format", ["xplane", "perfetto"])
def test_calibrate_without_step_windows_reports_no_launch_gap(
    tmp_path: Path, trace_format: str
) -> None:
    """Verify traces without step markers omit launch_gap and keep collectives."""
    op_duration_us = 10.0
    gap_us = 0.01
    event_specs = (
        ("op_compute_0", False),
        ("all-reduce.1", True),
        ("op_compute_1", False),
        ("all-reduce.1", True),
    )
    if trace_format == "xplane":
        xplane_events = [
            (
                op_name,
                idx * (op_duration_us + gap_us),
                op_duration_us,
                {"group_size": TP_GROUP_SIZE, "bytes": ALL_REDUCE_BYTES}
                if is_collective
                else {},
            )
            for idx, (op_name, is_collective) in enumerate(event_specs)
        ]
        trace_path = tmp_path / "no_steps.xplane.pb"
        trace_path.write_bytes(
            _build_xplane_bytes(lines=[("XLA Ops", 0, xplane_events)])
        )
    else:
        perfetto_events: list[dict[str, object]] = [
            {
                "ph": "X",
                "cat": "xla_op",
                "name": op_name,
                "pid": 1,
                "tid": 1,
                "ts": idx * (op_duration_us + gap_us),
                "dur": op_duration_us,
                "args": {"group_size": TP_GROUP_SIZE, "bytes": ALL_REDUCE_BYTES}
                if is_collective
                else {},
            }
            for idx, (op_name, is_collective) in enumerate(event_specs)
        ]
        trace_path = _write_json_trace(
            tmp_path,
            "no_steps_perfetto.json",
            _build_perfetto_trace(events=perfetto_events),
        )

    payload = advisor.calibrate_trace(trace_path)
    assert "launch_gap_us" not in payload
    assert len(payload["collectives"]) == 1
    assert payload["collectives"][0]["op"] == "all-reduce"
    assert payload["collectives"][0]["n"] == EXPECTED_PAIR_COUNT
    assert payload["collectives"][0]["latency_us"] == pytest.approx(op_duration_us)


def test_calibrate_reports_mean_collective_latency(tmp_path: Path) -> None:
    """Verify calibrated collective payload reports arithmetic mean latency."""
    collective_durations = (10.0, 10.0, 10.0, 20.0, 20.0)
    op_events: list[dict[str, object]] = [
        {
            "ph": "X",
            "cat": "xla_op",
            "name": "all-reduce",
            "pid": 1,
            "tid": 1,
            "ts": 50.0 + idx * 60.0,
            "dur": duration_us,
            "args": {"group_size": TP_GROUP_SIZE, "bytes": ALL_REDUCE_BYTES},
        }
        for idx, duration_us in enumerate(collective_durations)
    ]
    trace_file = _write_json_trace(
        tmp_path,
        "mean_latency_trace.json",
        _build_perfetto_trace(steps=[(0, "decode", 0, 500)], events=op_events),
    )
    payload = advisor.calibrate_trace(trace_file)
    assert len(payload["collectives"]) == 1
    assert payload["collectives"][0]["latency_us"] == pytest.approx(10.0)
    assert payload["collectives"][0]["mean_latency_us"] == pytest.approx(14.0)


def test_xla_modules_perfetto_propagates_host_tracks_and_clamps_overhang(
    tmp_path: Path,
) -> None:
    """Verify Perfetto XLA Modules propagates host tracks and clamps op overhangs."""
    perfetto_file = _write_json_trace(
        tmp_path,
        "xla_modules_overhang.json",
        _build_perfetto_trace(
            xla_module=_XlaModuleSpec(
                num_layers=XLA_MODULE_DEFAULT_LAYERS,
                ranks=(1, 2),
                host_wraps=True,
                overhanging_tail_us=500.0,
            )
        ),
    )
    _, _, _, _, active_span_us, _ = _xla_module_step_geometry(XLA_MODULE_DEFAULT_LAYERS)
    expected_host_head_us = XLA_MODULE_FIRST_OP_OFFSET_US - XLA_MODULE_START_OFFSET_US
    expected_steady_steps = XLA_MODULE_NUM_STEPS - 1
    expected_clamped_step_us = active_span_us + XLA_MODULE_POST_TAIL_US
    payload = advisor.calibrate_trace(
        perfetto_file,
        options=advisor.CalibrateOptions(
            chip="tpu7x",
            num_layers=XLA_MODULE_DEFAULT_LAYERS,
        ),
    )
    assert payload["host_head_us"]["p50"] == pytest.approx(
        expected_host_head_us, abs=1e-2
    )
    assert payload["host_head_us"]["n"] == EXPECTED_PAIR_COUNT * expected_steady_steps
    assert payload["step_time_us"]["p50"] == pytest.approx(
        expected_clamped_step_us, abs=1e-2
    )


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
@pytest.mark.parametrize("phase", ["decode", "prefill", "verify", "mixed"])
@pytest.mark.parametrize("trace_format", ["xplane", "perfetto"])
def test_calibrated_trace_round_trip_conservation_across_models_and_phases(
    tmp_path: Path,
    model_slug: str,
    phase: advisor.Phase,
    trace_format: str,
) -> None:
    """Verify calibrate -> load_measurements -> advise conserves across models."""
    profile = advisor.resolve_model_profile(model_slug)
    decode_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    num_layers = profile.num_layers
    boundaries = constants.SUBLAYER_BOUNDARIES_PER_LAYER * num_layers

    calibrated_by_scale: list[tuple[calibrate.CalibrationPayload, float, float]] = []
    for idle_scale in (1.0, 2.0):
        _, _, _, _, active_span_us, idle_gap_us = _xla_module_step_geometry(
            num_layers, idle_scale=idle_scale
        )
        spec = _XlaModuleSpec(num_layers=num_layers, idle_scale=idle_scale)
        if trace_format == "xplane":
            trace_file = tmp_path / f"{model_slug}_{phase}_{idle_scale}.xplane.pb"
            trace_file.write_bytes(_build_xplane_bytes(xla_module=spec))
        else:
            trace_file = _write_json_trace(
                tmp_path,
                f"{model_slug}_{phase}_{idle_scale}.json",
                _build_perfetto_trace(xla_module=spec),
            )
        out_json = tmp_path / f"{model_slug}_{phase}_{trace_format}_{idle_scale}.json"
        batch_size = (
            max(1, decode_layout.dp)
            if phase == "prefill"
            else specs.align_batch_to_dp(16, decode_layout.dp)
        )
        payload = advisor.calibrate_trace(
            trace_file,
            out_json,
            options=advisor.CalibrateOptions(
                chip="tpu7x",
                model=model_slug,
                phase=phase,
                batch=batch_size,
                tp=decode_layout.tp,
            ),
        )
        calibrated_by_scale.append((payload, active_span_us, idle_gap_us))

    payload_1x, active_span_us, idle_gap_1x_us = calibrated_by_scale[0]
    payload_2x, _, idle_gap_2x_us = calibrated_by_scale[1]
    expected_gap_1x_us = idle_gap_1x_us / boundaries
    expected_host_head_us = XLA_MODULE_FIRST_OP_OFFSET_US - XLA_MODULE_START_OFFSET_US

    assert payload_1x["launch_gap_us"]["unit"] == constants.CONSERVED_GAP_UNIT
    assert payload_1x["launch_gap_us"]["num_layers_norm"] == num_layers
    assert payload_1x["launch_gap_us"]["step_idle_gap_us"] == pytest.approx(
        idle_gap_1x_us, abs=1e-3
    )
    assert payload_1x["launch_gap_us"]["p50"] == pytest.approx(
        expected_gap_1x_us, abs=1e-6
    )
    assert payload_2x["launch_gap_us"]["step_idle_gap_us"] == pytest.approx(
        idle_gap_2x_us, abs=1e-3
    )
    assert payload_2x["launch_gap_us"]["p50"] / payload_1x["launch_gap_us"][
        "p50"
    ] == pytest.approx(2.0, rel=1e-6)

    out_1x_json = tmp_path / f"{model_slug}_{phase}_{trace_format}_1.0.json"
    calibrated = advisor.load_measurements(out_1x_json)
    calib_batch = (
        max(1, decode_layout.dp)
        if phase == "prefill"
        else specs.align_batch_to_dp(16, decode_layout.dp)
    )
    workload = build_preset_workload(
        model_slug,
        phase,
        use_decode_layout=True,
        batch=calib_batch,
        isl=4096,
        chunk_size=1024 if phase in ("prefill", "mixed") else 0,
        spec_tokens=(
            (decode_layout.spec_tokens or 3) if phase in ("verify", "mixed") else 0
        ),
        pcp=2 if phase in ("prefill", "mixed") else 1,
        layer_groups=(num_layers,),
    )
    report = advisor.advise(profile, workload, "tpu7x", measurements=calibrated)
    per_op_scope = next(scope for scope in report.scopes if scope.scope == "per_op")
    assert per_op_scope.ops_per_step == boundaries
    assert calibrated.launch_gap_p50_us is not None
    assert per_op_scope.ops_per_step * calibrated.launch_gap_p50_us == pytest.approx(
        idle_gap_1x_us, rel=1e-9, abs=1e-3
    )
    assert per_op_scope.launch_us == pytest.approx(idle_gap_1x_us, rel=1e-9, abs=1e-3)
    assert calibrated.step_time_us == pytest.approx(active_span_us, abs=1e-2)
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["per_op_step_us"]["tag"] == "MEASURED"
    assert report.metrics["per_op_step_us"]["value"] == pytest.approx(
        round(active_span_us, 2), abs=1e-2
    )
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["launch_gap_us"]["value"] == pytest.approx(
        expected_gap_1x_us, abs=1e-6
    )
    assert report.metrics["calibrated_host_head_us"]["tag"] == "MEASURED"
    assert report.metrics["calibrated_host_head_us"]["value"] == pytest.approx(
        expected_host_head_us, abs=1e-2
    )
    for scope_est in report.scopes:
        if scope_est.feasible:
            assert (
                scope_est.step_floor_strict_us
                <= scope_est.step_bound_us
                <= scope_est.step_achievable_us
                <= scope_est.step_achievable_upper_us
            )
            assert (
                scope_est.step_bound_us
                <= scope_est.step_upper_us
                <= scope_est.step_achievable_upper_us
            )


def test_calibrate_xplane_epoch_timestamps_preserve_submicrosecond_gaps(
    tmp_path: Path,
) -> None:
    """Verify epoch `timestamp_ns` subtraction keeps 0.05 us op gaps intact."""
    epoch_ns = 1_800_000_000_000_000_000
    expected_gap_us = 0.05
    op_duration_us = 10.0
    pb_path = tmp_path / "epoch_gaps.xplane.pb"
    pb_path.write_bytes(
        _build_xplane_bytes(
            lines=[
                (
                    "Steps",
                    epoch_ns,
                    [
                        ("Step", 0.0, 200.0, {"step_num": 0, "phase": "decode"}),
                        ("Step", 500.0, 60.0, {"step_num": 1, "phase": "decode"}),
                    ],
                ),
                (
                    "XLA Ops",
                    epoch_ns,
                    [
                        (
                            "op_compute",
                            505.0 + idx * (op_duration_us + expected_gap_us),
                            op_duration_us,
                            {},
                        )
                        for idx in range(4)
                    ],
                ),
                (
                    "Framework Name Scope",
                    epoch_ns,
                    [("jit(model_forward)", 505.0, 50.0, {})],
                ),
            ]
        )
    )
    payload = advisor.calibrate_trace(pb_path)
    expected_gap_count = 3
    assert payload["launch_gap_us"]["p50"] == pytest.approx(expected_gap_us, abs=1e-4)
    assert payload["launch_gap_us"]["p90"] == pytest.approx(expected_gap_us, abs=1e-4)
    assert payload["launch_gap_us"]["n"] == expected_gap_count


def test_calibrate_xstat_oneof_decodes_ref_bytes_and_zero_ints(
    tmp_path: Path,
) -> None:
    """Verify `XStat` oneofs, `XEventMetadata.stats` overlay, and coverage math."""
    pb_path = tmp_path / "oneof_stats.xplane.pb"
    pb_path.write_bytes(
        _build_xplane_bytes(
            extras=_XPlaneExtras(
                metadata_stats={
                    "all-reduce.static": {
                        "group_size": ("uint64", TP_GROUP_SIZE),
                        "bytes": ALL_REDUCE_BYTES,
                    }
                }
            ),
            lines=[
                (
                    "Steps",
                    0,
                    [
                        (
                            "Step",
                            200.0,
                            100.0,
                            {
                                "step_num": 1,
                                "phase": ("ref", "decode"),
                                "active_tokens": ("uint64", 16),
                                "fused_eligible_tokens": ("double", 8.0),
                            },
                        ),
                        (
                            "Step",
                            400.0,
                            100.0,
                            {
                                "step_num": 2,
                                "phase": ("bytes", b"prefill"),
                                "active_tokens": 16,
                                "fused_eligible_tokens": 0,
                            },
                        ),
                    ],
                ),
                (
                    "XLA Ops",
                    0,
                    [
                        ("all-reduce.static", 210.0, 12.0, {}),
                        ("all-reduce.static", 230.0, 14.0, {"bytes": 65536}),
                    ],
                ),
            ],
        )
    )
    decode_out = tmp_path / "oneof_decode.json"
    decode_payload = advisor.calibrate_trace(
        pb_path, decode_out, options=advisor.CalibrateOptions(phase="decode")
    )
    assert decode_payload["step_time_us"]["n"] == 1
    assert decode_payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(
        0.25
    )
    assert advisor.load_measurements(decode_out).step_coverage == pytest.approx(0.25)
    collectives_by_bytes = {
        item["bytes"]: item for item in decode_payload["collectives"]
    }
    assert set(collectives_by_bytes) == {ALL_REDUCE_BYTES, 65536}
    assert collectives_by_bytes[ALL_REDUCE_BYTES]["group_size"] == TP_GROUP_SIZE
    assert collectives_by_bytes[65536]["group_size"] == TP_GROUP_SIZE

    prefill_payload = advisor.calibrate_trace(
        pb_path, options=advisor.CalibrateOptions(phase="prefill")
    )
    assert prefill_payload["step_time_us"]["n"] == 1


def test_calibrate_conserved_gap_requires_model_or_num_layers(
    tmp_path: Path,
) -> None:
    """Verify `--model` and `--num-layers` match on XLA Modules even < 2 * L ops."""
    pb_path = tmp_path / "xla_mod_40_ops.xplane.pb"
    pb_path.write_bytes(
        _build_xplane_bytes(
            xla_module=_XlaModuleSpec(
                num_layers=XLA_MODULE_DEFAULT_LAYERS,
                num_ops_override=40,
            )
        )
    )
    with pytest.raises(errors.MegakernelConfigError, match="--model or --num-layers"):
        advisor.calibrate_trace(pb_path)

    for model_slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(model_slug)
        for num_ops in (40, 120):
            trace_pb = tmp_path / f"xla_mod_{model_slug}_{num_ops}.xplane.pb"
            trace_pb.write_bytes(
                _build_xplane_bytes(
                    xla_module=_XlaModuleSpec(
                        num_layers=profile.num_layers,
                        num_ops_override=num_ops,
                    )
                )
            )
            by_model = advisor.calibrate_trace(
                trace_pb, options=advisor.CalibrateOptions(model=model_slug)
            )
            out_layers = tmp_path / f"layers_{model_slug}_{num_ops}.json"
            exit_code = advisor.main(
                [
                    "calibrate",
                    "--trace",
                    str(trace_pb),
                    "--out",
                    str(out_layers),
                    "--num-layers",
                    str(profile.num_layers),
                ]
            )
            assert exit_code == 0
            by_layers = json.loads(out_layers.read_text(encoding="utf-8"))
            assert by_model["launch_gap_us"] == by_layers["launch_gap_us"]
            assert by_layers["launch_gap_us"]["num_layers_norm"] == profile.num_layers


def test_calibrate_pure_prefill_omits_zero_coverage_and_ignores_annotation_lines(
    tmp_path: Path,
) -> None:
    """Verify pure-prefill omits 0.0 coverage and device annotations skip."""
    prefill_pb = tmp_path / "pure_prefill_with_annotations.xplane.pb"
    prefill_pb.write_bytes(
        _build_xplane_bytes(
            lines=[
                (
                    "Steps",
                    0,
                    [
                        (
                            "Step",
                            0.0,
                            300.0,
                            {
                                "step_num": 0,
                                "phase": "prefill",
                                "active_tokens": 16,
                                "fused_eligible_tokens": 0,
                            },
                        ),
                        (
                            "Step",
                            500.0,
                            300.0,
                            {
                                "step_num": 1,
                                "phase": "prefill",
                                "active_tokens": 16,
                                "fused_eligible_tokens": 0,
                            },
                        ),
                    ],
                ),
                (
                    "XLA Ops",
                    0,
                    [
                        ("attn", 510.0, 100.0, {}),
                        ("mlp", 614.0, 100.0, {}),
                    ],
                ),
                ("Framework Name Scope", 0, [("jit(forward)", 505.0, 290.0, {})]),
                ("Source Code", 0, [("model.py:42", 505.0, 290.0, {})]),
                ("XLA TraceMe", 0, [("StepTrace", 505.0, 290.0, {})]),
                ("Framework Ops", 0, [("MatMul", 505.0, 290.0, {})]),
                ("Custom Aux Debug", 0, [("helper_span", 505.0, 290.0, {})]),
            ]
        )
    )
    prefill_out = tmp_path / "prefill_out.json"
    prefill_payload = advisor.calibrate_trace(
        prefill_pb,
        prefill_out,
        options=advisor.CalibrateOptions(phase="prefill"),
    )
    assert "step_coverage" not in prefill_payload
    assert advisor.load_measurements(prefill_out).step_coverage is None
    assert prefill_payload["launch_gap_us"]["p50"] == pytest.approx(4.0)
    assert prefill_payload["launch_gap_us"]["n"] == 1

    mixed_trace = _write_json_trace(
        tmp_path,
        "mixed_coverage.json",
        _build_perfetto_trace(
            events=[
                {
                    "ph": "X",
                    "name": "Step 1",
                    "cat": "step",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": 50.0,
                    "args": {
                        "phase": "decode",
                        "active_tokens": 16,
                        "fused_eligible_tokens": 16,
                    },
                },
                {
                    "ph": "X",
                    "name": "Step 2",
                    "cat": "step",
                    "pid": 1,
                    "tid": 1,
                    "ts": 200.0,
                    "dur": 50.0,
                    "args": {"phase": "prefill"},
                },
            ]
        ),
    )
    mixed_payload = advisor.calibrate_trace(
        mixed_trace, options=advisor.CalibrateOptions(phase="mixed")
    )
    assert mixed_payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(
        0.5
    )

    for bad_eligible, bad_active in ((20, 16), (-1, 16)):
        invalid_trace = _write_json_trace(
            tmp_path,
            f"invalid_tokens_{bad_eligible}.json",
            _build_perfetto_trace(
                events=[
                    {
                        "ph": "X",
                        "name": "Step 1",
                        "cat": "step",
                        "pid": 1,
                        "tid": 1,
                        "ts": 100.0,
                        "dur": 50.0,
                        "args": {
                            "phase": "decode",
                            "active_tokens": bad_active,
                            "fused_eligible_tokens": bad_eligible,
                        },
                    }
                ]
            ),
        )
        with pytest.raises(errors.MegakernelConfigError, match="token counts"):
            advisor.calibrate_trace(invalid_trace)


def test_calibrate_xla_modules_named_step_and_xla_ops_step_num_do_not_misclassify(
    tmp_path: Path,
) -> None:
    """Verify `step_*` XLA Modules trim host head and `XLA Ops` stay compute."""
    _, _, _, _, active_span_us, idle_gap_us = _xla_module_step_geometry(
        XLA_MODULE_DEFAULT_LAYERS
    )
    boundaries = constants.SUBLAYER_BOUNDARIES_PER_LAYER * XLA_MODULE_DEFAULT_LAYERS
    expected_host_head_us = XLA_MODULE_FIRST_OP_OFFSET_US - XLA_MODULE_START_OFFSET_US
    module_spec = _XlaModuleSpec(
        num_layers=XLA_MODULE_DEFAULT_LAYERS,
        module_name="step_forward_jit",
        module_step_num=True,
    )

    for trace_path in (
        tmp_path / "step_named_module.xplane.pb",
        tmp_path / "step_named_module.json",
    ):
        if trace_path.suffix == ".pb":
            trace_path.write_bytes(_build_xplane_bytes(xla_module=module_spec))
        else:
            _write_json_trace(
                tmp_path,
                trace_path.name,
                _build_perfetto_trace(xla_module=module_spec),
            )
        payload = advisor.calibrate_trace(
            trace_path,
            options=advisor.CalibrateOptions(
                chip="tpu7x",
                num_layers=XLA_MODULE_DEFAULT_LAYERS,
            ),
        )
        assert payload["host_head_us"]["p50"] == pytest.approx(
            expected_host_head_us, abs=1e-2
        )
        assert payload["step_time_us"]["p50"] == pytest.approx(active_span_us, abs=1e-2)
        assert payload["launch_gap_us"]["p50"] == pytest.approx(
            idle_gap_us / boundaries, abs=1e-6
        )

    xops_pb = tmp_path / "xla_ops_step_num.xplane.pb"
    xops_pb.write_bytes(
        _build_xplane_bytes(
            lines=[
                (
                    "Steps",
                    0,
                    [("Step", 0.0, 100.0, {"step_num": 1, "phase": "decode"})],
                ),
                (
                    "XLA Ops",
                    0,
                    [
                        ("step_embed", 10.0, 20.0, {"step_num": 1}),
                        ("attn_proj", 33.0, 20.0, {"step_num": 1}),
                    ],
                ),
            ]
        )
    )
    xops_payload = advisor.calibrate_trace(xops_pb)
    assert xops_payload["step_time_us"]["n"] == 1
    assert xops_payload["launch_gap_us"]["p50"] == pytest.approx(3.0)
    assert xops_payload["launch_gap_us"]["n"] == 1


def test_calibrate_nested_events_and_aggregated_num_occurrences(
    tmp_path: Path,
) -> None:
    """Verify nested spans track running `max_end_us` and `num_occurrences` skips."""
    pb_path = tmp_path / "nested_and_aggregated.xplane.pb"
    pb_path.write_bytes(
        _build_xplane_bytes(
            lines=[
                (
                    "Steps",
                    0,
                    [("Step", 0.0, 200.0, {"step_num": 1, "phase": "decode"})],
                ),
                (
                    "XLA Ops",
                    0,
                    [
                        ("outer_fusion", 0.0, 100.0, {}),
                        ("inner_a", 10.0, 10.0, {}),
                        ("inner_b", 30.0, 10.0, {}),
                        ("tail_op", 102.0, 8.0, {}),
                    ],
                ),
            ],
            extras=_XPlaneExtras(
                num_occurrences_events=[("XLA Ops", "all-reduce.agg", 50, 500.0)]
            ),
        )
    )
    payload = advisor.calibrate_trace(pb_path, options=advisor.CalibrateOptions(tp=8))
    assert payload["launch_gap_us"]["n"] == 1
    assert payload["launch_gap_us"]["p50"] == pytest.approx(2.0)
    assert payload["collectives"] == []


def test_calibrate_perfetto_string_pids_epoch_ts_and_unicode_digits(
    tmp_path: Path,
) -> None:
    """Verify string PIDs, epoch Perfetto timestamps, and Unicode digits parse."""
    raw_json_text = (
        '{"traceEvents": ['
        '{"ph":"X","cat":"step","name":"Step","pid":"TPU:0","tid":1,'
        '"ts":1800000000000000.00,"dur":500.0,'
        '"args":{"step_num":"1","phase":"decode","active_tokens":"²"}},'
        '{"ph":"X","cat":"step","name":"Step","pid":"TPU:1","tid":1,'
        '"ts":1800000000000000.00,"dur":500.0,'
        '"args":{"step_num":"1","phase":"decode"}},'
        '{"ph":"X","cat":"xla_op","name":"op_0","pid":"TPU:0","tid":"stream_a",'
        '"ts":1800000000000010.00,"dur":10.0},'
        '{"ph":"X","cat":"xla_op","name":"op_1","pid":"TPU:0","tid":"stream_a",'
        '"ts":1800000000000020.05,"dur":10.0},'
        '{"ph":"X","cat":"xla_op","name":"op_2","pid":"TPU:1","tid":"stream_a",'
        '"ts":1800000000000030.00,"dur":10.0},'
        '{"ph":"X","cat":"xla_op","name":"op_3","pid":"TPU:1","tid":"stream_a",'
        '"ts":1800000000000045.05,"dur":10.0}'
        "]}"
    )
    trace_file = tmp_path / "epoch_string_pids.json"
    trace_file.write_text(raw_json_text, encoding="utf-8")
    payload = advisor.calibrate_trace(trace_file)
    assert payload["launch_gap_us"]["n"] == EXPECTED_PAIR_COUNT
    assert payload["launch_gap_us"]["p50"] == pytest.approx(2.55, abs=1e-4)
    assert payload["launch_gap_us"]["p90"] == pytest.approx(4.55, abs=1e-4)


def test_calibrate_host_head_matches_filtered_phase(tmp_path: Path) -> None:
    """Verify `host_head_us` aligns with phase-filtered `Steps` and steady ranking."""
    pb_path = tmp_path / "mixed_steps_and_modules.xplane.pb"
    pb_path.write_bytes(
        _build_xplane_bytes(
            lines=[
                (
                    "Steps",
                    0,
                    [
                        ("Step", 100.0, 500.0, {"step_num": 1, "phase": "prefill"}),
                        ("Step", 1010.0, 200.0, {"step_num": 2, "phase": "decode"}),
                        ("Step", 2010.0, 200.0, {"step_num": 3, "phase": "decode"}),
                    ],
                ),
                (
                    "XLA Modules",
                    0,
                    [
                        ("jit_prefill", 0.0, 600.0, {}),
                        ("jit_decode", 1000.0, 210.0, {}),
                        ("jit_decode", 2000.0, 210.0, {}),
                    ],
                ),
                (
                    "XLA Ops",
                    0,
                    [
                        ("prefill_op", 100.0, 500.0, {}),
                        ("decode_op_1", 1010.0, 200.0, {}),
                        ("decode_op_2", 2010.0, 200.0, {}),
                    ],
                ),
            ]
        )
    )
    decode_payload = advisor.calibrate_trace(
        pb_path, options=advisor.CalibrateOptions(phase="decode")
    )
    assert decode_payload["step_time_us"]["p50"] == pytest.approx(200.0)
    assert decode_payload["host_head_us"]["p50"] == pytest.approx(10.0)
    assert decode_payload["host_head_us"]["n"] == EXPECTED_PAIR_COUNT

    rank_pb = tmp_path / "steady_program_ranking.xplane.pb"
    rank_pb.write_bytes(
        _build_xplane_bytes(
            lines=[
                (
                    "XLA Modules",
                    0,
                    [
                        ("jit_alpha_heavy_warmup", 0.0, 10_000.0, {}),
                        ("jit_alpha_heavy_warmup", 15_000.0, 100.0, {}),
                        ("jit_beta_steady", 20_000.0, 500.0, {}),
                        ("jit_beta_steady", 21_000.0, 500.0, {}),
                        ("jit_beta_steady", 22_000.0, 500.0, {}),
                    ],
                ),
                (
                    "XLA Ops",
                    0,
                    [
                        ("alpha_w", 200.0, 9_800.0, {}),
                        ("alpha_s", 15_010.0, 90.0, {}),
                        ("beta_0", 20_010.0, 490.0, {}),
                        ("beta_1", 21_010.0, 490.0, {}),
                        ("beta_2", 22_010.0, 490.0, {}),
                    ],
                ),
            ]
        )
    )
    rank_payload = advisor.calibrate_trace(
        rank_pb, options=advisor.CalibrateOptions(num_layers=1)
    )
    assert rank_payload["step_time_us"]["p50"] == pytest.approx(490.0)
    assert rank_payload["host_head_us"]["p50"] == pytest.approx(10.0)


def test_calibrate_xplane_per_plane_origin_preserves_mixed_epoch_precision(
    tmp_path: Path,
) -> None:
    """Verify per-plane `origin_ns` isolates epoch precision from relative files."""
    epoch_ns = 1_800_000_000_000_000_000
    expected_gap_us = 0.05
    op_duration_us = 10.0
    rel_bytes = _build_xplane_bytes(
        lines=[
            (
                "Steps",
                50_000,
                [("Step", 0.0, 100.0, {"step_num": 1, "phase": "decode"})],
            ),
            (
                "XLA Ops",
                50_000,
                [
                    ("rel_a", 10.0, op_duration_us, {}),
                    (
                        "rel_b",
                        10.0 + op_duration_us + expected_gap_us,
                        op_duration_us,
                        {},
                    ),
                ],
            ),
        ]
    )
    epoch_bytes = _build_xplane_bytes(
        lines=[
            (
                "Steps",
                epoch_ns,
                [("Step", 0.0, 100.0, {"step_num": 1, "phase": "decode"})],
            ),
            (
                "XLA Ops",
                epoch_ns,
                [
                    ("epoch_a", 10.0, op_duration_us, {}),
                    (
                        "epoch_b",
                        10.0 + op_duration_us + expected_gap_us,
                        op_duration_us,
                        {},
                    ),
                ],
            ),
        ]
    )
    trace_dir = _write_xplane_dir(
        tmp_path / "mixed_origins",
        {"rank0.xplane.pb": rel_bytes, "rank1.xplane.pb": epoch_bytes},
    )
    payload = advisor.calibrate_trace(trace_dir)
    assert payload["launch_gap_us"]["n"] == EXPECTED_PAIR_COUNT
    assert payload["launch_gap_us"]["p50"] == pytest.approx(expected_gap_us, abs=1e-4)
    assert payload["launch_gap_us"]["p90"] == pytest.approx(expected_gap_us, abs=1e-4)


def _shifted_perfetto_events(shift_us: int) -> list[dict[str, object]]:
    """Build a 2-step Perfetto trace shifted by `shift_us` with a 400 us host head."""
    host_head_us = 400.0
    op_dur_us = 10.0
    gap_us = 3.0
    step_dur_us = host_head_us + 2.0 * op_dur_us + gap_us + 5.0
    step_starts = (0.0, 1000.0)
    events: list[dict[str, object]] = []
    for step_idx, step_start in enumerate(step_starts):
        base_ts = shift_us + step_start
        events.append(
            {
                "name": f"xla_module_{step_idx}",
                "cat": "xla_module",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": base_ts,
                "dur": step_dur_us,
            }
        )
        first_op_ts = base_ts + host_head_us
        events.append(
            {
                "name": "qkv_proj",
                "cat": "op",
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": first_op_ts,
                "dur": op_dur_us,
            }
        )
        events.append(
            {
                "name": "all-reduce.1",
                "cat": "op",
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": first_op_ts + op_dur_us + gap_us,
                "dur": op_dur_us,
                "args": {"group_size": TP_GROUP_SIZE, "bytes": ALL_REDUCE_BYTES},
            }
        )
    return events


def test_calibrate_timeline_shift_and_empty_xline_invariance(tmp_path: Path) -> None:
    """Verify Perfetto shifts and empty XPlane sibling lines preserve timing."""
    # Tolerance is 1 ulp at QUANTILE_DECIMALS = 4.
    tol_us = 10.0 ** (-constants.QUANTILE_DECIMALS)
    baseline_path = _write_json_trace(
        tmp_path, "shift_1000.json", {"traceEvents": _shifted_perfetto_events(1000)}
    )
    shift_opts = advisor.CalibrateOptions(num_layers=1)
    baseline = advisor.calibrate_trace(baseline_path, options=shift_opts)
    expected_gap_us = 3.0 / constants.SUBLAYER_BOUNDARIES_PER_LAYER
    expected_host_head_us = 400.0
    assert baseline["launch_gap_us"]["p50"] == pytest.approx(
        expected_gap_us, abs=tol_us
    )
    assert baseline["host_head_us"]["p50"] == pytest.approx(
        expected_host_head_us, abs=tol_us
    )
    for shift_us in (-20, 0, 1_800_000_000_000):
        trace_path = _write_json_trace(
            tmp_path,
            f"shift_{shift_us}.json",
            {"traceEvents": _shifted_perfetto_events(shift_us)},
        )
        shifted = advisor.calibrate_trace(trace_path, options=shift_opts)
        assert shifted["launch_gap_us"]["p50"] == pytest.approx(
            baseline["launch_gap_us"]["p50"], abs=tol_us
        )
        assert shifted["host_head_us"]["p50"] == pytest.approx(
            baseline["host_head_us"]["p50"], abs=tol_us
        )
        assert shifted["step_time_us"]["p50"] == pytest.approx(
            baseline["step_time_us"]["p50"], abs=tol_us
        )
        assert shifted["collectives"] == baseline["collectives"]

    epoch_ns = 1_800_000_000_000_000_000
    expected_xplane_gap_us = 0.05
    op_dur_us = 10.0
    xplane_path = tmp_path / "empty_and_zero_sibling.xplane.pb"
    xplane_path.write_bytes(
        _build_xplane_bytes(
            lines=[
                ("EmptyControlLine", 1000, []),
                ("ZeroRelativeLine", 0, [("idle_marker", 0.0, 1.0, {})]),
                (
                    "Steps",
                    epoch_ns,
                    [
                        ("Step", 0.0, 200.0, {"step_num": 0, "phase": "decode"}),
                        ("Step", 500.0, 60.0, {"step_num": 1, "phase": "decode"}),
                    ],
                ),
                (
                    "XLA Ops",
                    epoch_ns,
                    [
                        (
                            "op_compute",
                            505.0 + idx * (op_dur_us + expected_xplane_gap_us),
                            op_dur_us,
                            {},
                        )
                        for idx in range(4)
                    ],
                ),
            ]
        )
    )
    xplane_payload = advisor.calibrate_trace(xplane_path)
    assert xplane_payload["launch_gap_us"]["p50"] == pytest.approx(
        expected_xplane_gap_us, abs=tol_us
    )


def test_calibrate_active_span_conservation_excludes_host_head_and_tail(
    tmp_path: Path,
) -> None:
    """Verify explicit Steps conserved idle gap equals active span minus union busy."""
    norm_layers = 12
    boundaries = constants.SUBLAYER_BOUNDARIES_PER_LAYER * norm_layers
    op_count = boundaries
    op_dur_us = 10.0
    intra_gap_us = 2.0
    expected_idle_us = (op_count - 1) * intra_gap_us
    expected_per_boundary_us = expected_idle_us / boundaries
    gap_tol_us = 10.0 ** (-constants.QUANTILE_DECIMALS)
    boundary_tol_us = 10.0 ** (-constants.CONSERVED_GAP_DECIMALS)

    for host_head_us, host_tail_us in ((0.0, 0.0), (500.0, 0.0), (500.0, 1200.0)):
        active_span_us = op_count * op_dur_us + expected_idle_us
        step_dur_us = host_head_us + active_span_us + host_tail_us
        events: list[dict[str, object]] = []
        for step_idx in range(3):
            step_start = 100.0 + step_idx * (step_dur_us + 500.0)
            events.append(
                {
                    "name": f"step_{step_idx}",
                    "cat": "step",
                    "ph": "X",
                    "pid": 1,
                    "tid": 0,
                    "ts": step_start,
                    "dur": step_dur_us,
                    "args": {"phase": "decode"},
                }
            )
            events.extend(
                {
                    "name": f"gemm_{op_idx}",
                    "cat": "op",
                    "ph": "X",
                    "pid": 1,
                    "tid": 1,
                    "ts": step_start
                    + host_head_us
                    + op_idx * (op_dur_us + intra_gap_us),
                    "dur": op_dur_us,
                }
                for op_idx in range(op_count)
            )
        trace_path = _write_json_trace(
            tmp_path,
            f"steps_head_{int(host_head_us)}_tail_{int(host_tail_us)}.json",
            {"traceEvents": events},
        )
        payload = advisor.calibrate_trace(
            trace_path, options=advisor.CalibrateOptions(num_layers=norm_layers)
        )
        assert payload["launch_gap_us"]["unit"] == constants.CONSERVED_GAP_UNIT
        assert payload["launch_gap_us"]["step_idle_gap_us"] == pytest.approx(
            expected_idle_us, abs=gap_tol_us
        )
        assert payload["launch_gap_us"]["p50"] == pytest.approx(
            expected_per_boundary_us, abs=boundary_tol_us
        )


def test_calibrate_xplane_perfetto_event_classification_parity(tmp_path: Path) -> None:
    """Verify `step_*` compute ops with `step_num` never promote to step windows."""
    op_dur_us = 10.0
    gap_us = 4.0
    tol_us = 10.0 ** (-constants.QUANTILE_DECIMALS)

    for xplane_line, perfetto_cat in (
        ("XLA Ops", "op"),
        ("Async XLA Ops", "xla_op"),
    ):
        xplane_path = tmp_path / f"parity_{perfetto_cat}.xplane.pb"
        xplane_path.write_bytes(
            _build_xplane_bytes(
                lines=[
                    (
                        "Steps",
                        1_000_000,
                        [
                            ("Step", 0.0, 200.0, {"step_num": 0, "phase": "decode"}),
                            ("Step", 500.0, 100.0, {"step_num": 1, "phase": "decode"}),
                            ("Step", 900.0, 100.0, {"step_num": 2, "phase": "decode"}),
                        ],
                    ),
                    (
                        xplane_line,
                        1_000_000,
                        [
                            ("step_gemm_0", 510.0, op_dur_us, {"step_num": 1}),
                            (
                                "all-reduce.1",
                                510.0 + op_dur_us + gap_us,
                                op_dur_us,
                                {
                                    "step_num": 1,
                                    "group_size": TP_GROUP_SIZE,
                                    "bytes": ALL_REDUCE_BYTES,
                                },
                            ),
                            ("step_gemm_2", 910.0, op_dur_us, {"step_num": 2}),
                            (
                                "all-reduce.2",
                                910.0 + op_dur_us + gap_us,
                                op_dur_us,
                                {
                                    "step_num": 2,
                                    "group_size": TP_GROUP_SIZE,
                                    "bytes": ALL_REDUCE_BYTES,
                                },
                            ),
                        ],
                    ),
                ]
            )
        )
        perfetto_events: list[dict[str, object]] = [
            {
                "name": "Step",
                "cat": "step",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 1000.0,
                "dur": 200.0,
                "args": {"step_num": 0, "phase": "decode"},
            },
            {
                "name": "Step",
                "cat": "step",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 1500.0,
                "dur": 100.0,
                "args": {"step_num": 1, "phase": "decode"},
            },
            {
                "name": "Step",
                "cat": "step",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": 1900.0,
                "dur": 100.0,
                "args": {"step_num": 2, "phase": "decode"},
            },
            {
                "name": "step_gemm_0",
                "cat": perfetto_cat,
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": 1510.0,
                "dur": op_dur_us,
                "args": {"step_num": 1},
            },
            {
                "name": "all-reduce.1",
                "cat": perfetto_cat,
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": 1510.0 + op_dur_us + gap_us,
                "dur": op_dur_us,
                "args": {
                    "step_num": 1,
                    "group_size": TP_GROUP_SIZE,
                    "bytes": ALL_REDUCE_BYTES,
                },
            },
            {
                "name": "step_gemm_2",
                "cat": perfetto_cat,
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": 1910.0,
                "dur": op_dur_us,
                "args": {"step_num": 2},
            },
            {
                "name": "all-reduce.2",
                "cat": perfetto_cat,
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": 1910.0 + op_dur_us + gap_us,
                "dur": op_dur_us,
                "args": {
                    "step_num": 2,
                    "group_size": TP_GROUP_SIZE,
                    "bytes": ALL_REDUCE_BYTES,
                },
            },
        ]
        perfetto_path = _write_json_trace(
            tmp_path, f"parity_{perfetto_cat}.json", {"traceEvents": perfetto_events}
        )
        xplane_out = advisor.calibrate_trace(xplane_path)
        perfetto_out = advisor.calibrate_trace(perfetto_path)
        assert xplane_out["launch_gap_us"]["p50"] == pytest.approx(gap_us, abs=tol_us)
        assert perfetto_out["launch_gap_us"] == xplane_out["launch_gap_us"]
        assert perfetto_out["collectives"] == xplane_out["collectives"]
        assert perfetto_out["step_time_us"] == xplane_out["step_time_us"]


@dataclasses.dataclass(frozen=True)
class _Pp2TraceSpec:
    """Synthetic pp=2 trace geometry for round-trip calibration verification."""

    model_slug: str
    phase: str
    trace_format: str
    boundaries: int
    tp_degree: int
    phase_ar_bytes: int
    op_dur_us: float
    gap_us: float
    expected_idle_us: float


def _write_pp2_round_trip_trace(tmp_path: Path, spec: _Pp2TraceSpec) -> Path:
    """Write a 3-step XPlane or Perfetto trace for pp=2 round-trip verification."""
    step_span_us = spec.boundaries * spec.op_dur_us + spec.expected_idle_us
    if spec.trace_format == "xplane":
        op_events: list[tuple[str, float, float, dict[str, object]]] = [
            (
                "all-reduce.1" if op_idx == 1 else f"op_{op_idx}",
                100.0 + step_idx * 5000.0 + op_idx * (spec.op_dur_us + spec.gap_us),
                spec.op_dur_us,
                (
                    {"group_size": spec.tp_degree, "bytes": spec.phase_ar_bytes}
                    if op_idx == 1
                    else {}
                ),
            )
            for step_idx in range(3)
            for op_idx in range(spec.boundaries)
        ]
        trace_path = tmp_path / f"{spec.model_slug}_{spec.phase}_pp2.xplane.pb"
        trace_path.write_bytes(
            _build_xplane_bytes(
                lines=[
                    (
                        "Steps",
                        1_000_000,
                        [
                            (
                                "Step",
                                100.0 + step_idx * 5000.0,
                                step_span_us,
                                {"step_num": step_idx, "phase": spec.phase},
                            )
                            for step_idx in range(3)
                        ],
                    ),
                    ("XLA Ops", 1_000_000, op_events),
                ]
            )
        )
        return trace_path

    perf_events: list[dict[str, object]] = []
    for step_idx in range(3):
        step_start = 100.0 + step_idx * 5000.0
        perf_events.append(
            {
                "name": f"step_{step_idx}",
                "cat": "step",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": step_start,
                "dur": step_span_us,
                "args": {"phase": spec.phase},
            }
        )
        perf_events.extend(
            {
                "name": "all-reduce.1" if op_idx == 1 else f"op_{op_idx}",
                "cat": "op",
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": step_start + op_idx * (spec.op_dur_us + spec.gap_us),
                "dur": spec.op_dur_us,
                "args": (
                    {"group_size": spec.tp_degree, "bytes": spec.phase_ar_bytes}
                    if op_idx == 1
                    else {}
                ),
            }
            for op_idx in range(spec.boundaries)
        )
    return _write_json_trace(
        tmp_path,
        f"{spec.model_slug}_{spec.phase}_pp2.json",
        {"traceEvents": perf_events},
    )


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
@pytest.mark.parametrize("trace_format", ["xplane", "perfetto"])
def test_calibrated_round_trip_and_strict_step_time_across_presets(
    tmp_path: Path,
    model_slug: str,
    trace_format: str,
) -> None:
    """Verify pp>1 norm, all-reduce byte matching, and sub-50ps strict step time."""
    profile = advisor.resolve_model_profile(model_slug)
    decode_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    pp_degree = 2
    per_rank_layers = math.ceil(profile.num_layers / pp_degree)
    boundaries = constants.SUBLAYER_BOUNDARIES_PER_LAYER * per_rank_layers
    batch_size = 16
    act_bytes = constants.DTYPE_BYTES[profile.act_dtype]
    ar_bytes = round(profile.hidden_size * batch_size * act_bytes)
    op_dur_us = 10.0
    gap_us = 2.0
    expected_idle_us = (boundaries - 1) * gap_us
    band_tol_us = 2.0 * (0.5 * 10.0 ** (-constants.QUANTILE_DECIMALS + 2))
    gap_tol_us = 10.0 ** (-constants.QUANTILE_DECIMALS)

    for phase in ("decode", "prefill", "verify", "mixed"):
        phase_batch = (
            decode_layout.dp
            if phase == "prefill"
            else specs.align_batch_to_dp(batch_size, decode_layout.dp)
        )
        phase_ar_bytes = round(profile.hidden_size * phase_batch * act_bytes)
        trace_path = _write_pp2_round_trip_trace(
            tmp_path,
            _Pp2TraceSpec(
                model_slug=model_slug,
                phase=phase,
                trace_format=trace_format,
                boundaries=boundaries,
                tp_degree=decode_layout.tp,
                phase_ar_bytes=phase_ar_bytes,
                op_dur_us=op_dur_us,
                gap_us=gap_us,
                expected_idle_us=expected_idle_us,
            ),
        )
        out_json = tmp_path / f"{model_slug}_{phase}_{trace_format}_pp2.json"
        payload = advisor.calibrate_trace(
            trace_path,
            out_json,
            options=advisor.CalibrateOptions(
                chip="tpu7x",
                model=model_slug,
                phase=phase,
                batch=phase_batch,
                tp=decode_layout.tp,
                num_layers=per_rank_layers,
            ),
        )
        assert payload["launch_gap_us"]["num_layers_norm"] == per_rank_layers
        assert payload["launch_gap_us"]["step_idle_gap_us"] == pytest.approx(
            expected_idle_us, abs=gap_tol_us
        )
        loaded = advisor.load_measurements(out_json)
        workload = build_preset_workload(
            model_slug,
            phase,
            use_decode_layout=True,
            batch=phase_batch,
            isl=constants.DEFAULT_PREFILL_ISL,
            chunk_size=1024 if phase in ("prefill", "mixed") else 0,
            spec_tokens=(
                (decode_layout.spec_tokens or constants.DEFAULT_SPEC_TOKENS)
                if phase in ("verify", "mixed")
                else 0
            ),
            pp=pp_degree,
            pcp=1,
        )
        report = advisor.advise(profile, workload, "tpu7x", measurements=loaded)
        per_op_scope = next(s for s in report.scopes if s.scope == "per_op")
        assert per_op_scope.launch_us == pytest.approx(
            expected_idle_us, abs=band_tol_us
        )
        if phase == "decode" and decode_layout.tp > 1:
            assert ar_bytes == phase_ar_bytes
            assert report.metrics["collective_call_us_xla"]["tag"] == "MEASURED"
        for scope_est in report.scopes:
            if scope_est.feasible:
                assert (
                    scope_est.step_floor_strict_us
                    <= scope_est.step_bound_us + band_tol_us
                )
                assert (
                    scope_est.step_bound_us
                    <= scope_est.step_achievable_us + band_tol_us
                )
                assert (
                    scope_est.step_achievable_us
                    <= scope_est.step_achievable_upper_us + band_tol_us
                )

    # Sub-50ps steps must preserve positive step_time_us for strict load_measurements.
    sub_50ps_us = 1e-5
    tiny_trace = _write_json_trace(
        tmp_path,
        f"{model_slug}_{trace_format}_sub50ps.json",
        {
            "traceEvents": [
                {
                    "name": "step_1",
                    "cat": "step",
                    "ph": "X",
                    "pid": 1,
                    "tid": 0,
                    "ts": 100.0,
                    "dur": sub_50ps_us,
                    "args": {"phase": "decode"},
                },
                {
                    "name": "tiny_op",
                    "cat": "op",
                    "ph": "X",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": sub_50ps_us,
                },
            ]
        },
    )
    tiny_out = tmp_path / f"{model_slug}_{trace_format}_sub50ps_out.json"
    tiny_payload = advisor.calibrate_trace(tiny_trace, tiny_out)
    assert tiny_payload["step_time_us"]["p50"] > 0.0
    assert tiny_payload["step_time_us"]["p90"] > 0.0
    tiny_loaded = advisor.load_measurements(tiny_out)
    assert tiny_loaded.step_time_us is not None
    assert tiny_loaded.step_time_us > 0.0


def test_calibrate_fail_closed_parsing_and_cli_empty_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify negative stats, padded phase, XPlane unavailable, and CLI guards."""
    neg_stat_trace = _write_json_trace(
        tmp_path,
        "neg_stat.json",
        {
            "traceEvents": [
                {
                    "name": "step_1",
                    "cat": "step",
                    "ph": "X",
                    "pid": 1,
                    "tid": 0,
                    "ts": 100.0,
                    "dur": 50.0,
                    "args": {
                        "phase": "decode",
                        "active_tokens": "16",
                        "fused_eligible_tokens": "-5",
                    },
                }
            ]
        },
    )
    with pytest.raises(errors.MegakernelConfigError, match="Invalid step token counts"):
        advisor.calibrate_trace(neg_stat_trace)

    padded_phase_trace = _write_json_trace(
        tmp_path,
        "padded_phase.json",
        {
            "traceEvents": [
                {
                    "name": "step_1",
                    "cat": "step",
                    "ph": "X",
                    "pid": 1,
                    "tid": 0,
                    "ts": 100.0,
                    "dur": 50.0,
                    "args": {"phase": "  decode  "},
                },
                {
                    "name": "op_a",
                    "cat": "op",
                    "ph": "X",
                    "pid": 1,
                    "tid": 1,
                    "ts": 105.0,
                    "dur": 10.0,
                },
            ]
        },
    )
    padded_payload = advisor.calibrate_trace(
        padded_phase_trace, options=advisor.CalibrateOptions(phase="decode")
    )
    assert padded_payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(
        1.0
    )

    pb_path = tmp_path / "dummy.xplane.pb"
    pb_path.write_bytes(b"not_read_when_missing_pb2")
    with monkeypatch.context() as patched:
        patched.setattr(calibrate, "xplane_pb2", None)
        with pytest.raises(advisor.XPlaneToolUnavailableError):
            advisor.calibrate_trace(pb_path)

    cwd_dir = tmp_path / "cwd_clean"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)
    valid_trace = padded_phase_trace
    valid_out = tmp_path / "valid_out.json"
    bad_cli_cases: tuple[list[str], ...] = (
        ["calibrate", "--trace", "", "--out", str(valid_out)],
        ["calibrate", "--trace", "   ", "--out", str(valid_out)],
        ["calibrate", "--trace", str(valid_trace), "--out", ""],
        ["calibrate", "--trace", str(valid_trace), "--out", "   "],
        ["calibrate", "--trace", str(valid_trace), "--out", str(cwd_dir)],
        ["calibrate", "--trace", str(valid_trace), "--out", "-"],
        [
            "calibrate",
            "--trace",
            str(valid_trace),
            "--out",
            str(valid_out),
            "--chip",
            "",
        ],
        [
            "calibrate",
            "--trace",
            str(valid_trace),
            "--out",
            str(valid_out),
            "--model",
            "",
        ],
    )
    for argv in bad_cli_cases:
        with pytest.raises(errors.MegakernelConfigError):
            advisor.main(argv)
    assert list(cwd_dir.iterdir()) == []


def test_calibrate_cli_matches_api_payload(tmp_path: Path) -> None:
    """Verify `advisor calibrate` CLI JSON output matches `calibrate_trace` directly."""
    perfetto_path = _write_json_trace(
        tmp_path,
        "cli_parity.json",
        _build_perfetto_trace(synthetic_default=True),
    )
    xplane_path = tmp_path / "cli_parity.xplane.pb"
    xplane_path.write_bytes(_build_xplane_bytes(synthetic_default=True))
    for idx, trace_path in enumerate((perfetto_path, xplane_path)):
        out_json = tmp_path / f"cli_out_{idx}.json"
        exit_code = advisor.main(
            [
                "calibrate",
                "--trace",
                str(trace_path),
                "--out",
                str(out_json),
                "--chip",
                "tpu7x",
                "--model",
                "qwen3.5-397b",
                "--phase",
                "decode",
                "--tp",
                str(TP_GROUP_SIZE),
            ]
        )
        assert exit_code == 0
        cli_payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert "launch_gap_us" in cli_payload
        api_payload = advisor.calibrate_trace(
            trace_path,
            options=advisor.CalibrateOptions(
                chip="tpu7x",
                model="qwen3.5-397b",
                phase="decode",
                tp=TP_GROUP_SIZE,
            ),
        )
        assert cli_payload == api_payload
