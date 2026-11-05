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

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vllm_torchtpu.megakernel import advisor, errors

if TYPE_CHECKING:
    from typing import Protocol

    class _MutableXStatProto(Protocol):
        """Mutable XStat protobuf message protocol for test fixture construction."""

        metadata_id: int
        int64_value: int
        double_value: float
        str_value: str

    class _StatContainerProto(Protocol):
        """Repeated XStat container protocol."""

        def add(self) -> _MutableXStatProto:
            """Append and return a new XStat entry."""

    class _MutableXEventProto(Protocol):
        """Mutable XEvent protobuf message protocol for test fixture construction."""

        metadata_id: int
        offset_ps: int
        duration_ps: int
        stats: _StatContainerProto

    class _EventContainerProto(Protocol):
        """Repeated XEvent container protocol."""

        def add(self) -> _MutableXEventProto:
            """Append and return a new XEvent entry."""

    class _MutableXLineProto(Protocol):
        """Mutable XLine protobuf message protocol for test fixture construction."""

        id: int
        name: str
        timestamp_ns: int
        events: _EventContainerProto


_PERFETTO_FIXTURE = (
    Path(__file__).resolve().parent / "data" / "advisor" / "synthetic_perfetto.json.gz"
)
_PS_PER_US = 1_000_000
_EXPECTED_SINGLE_GAPS = 12
_EXPECTED_MULTI_GAPS = 24
_EXPECTED_GROUP_SIZE = 8
_EXPECTED_FALLBACK_TP = 4
_EXPECTED_AR_BYTES = 32768
_EXPECTED_A2A_BYTES = 65536
_EXPECTED_STEADY_STEPS = 4
_EXPECTED_BATCH = 16
_EXPECTED_TP = 8


def _add_int_stat(ev: _MutableXEventProto, meta_id: int, value: int) -> None:
    """Append an int64 XStat entry to `ev`."""
    st = ev.stats.add()
    st.metadata_id = meta_id
    st.int64_value = value


def _add_str_stat(ev: _MutableXEventProto, meta_id: int, value: str) -> None:
    """Append a string XStat entry to `ev`."""
    st = ev.stats.add()
    st.metadata_id = meta_id
    st.str_value = value


def _append_steady_step_events(
    step_line: _MutableXLineProto,
    ops_line: _MutableXLineProto,
    step_spec: tuple[int, int, int, str, int, int],
    time_shift_us: int,
) -> None:
    """Append one steady-state step and its 4 compute ops to `step_line`/`ops_line`."""
    step_num, base_us, dur_us, phase_str, act_tok, elig_tok = step_spec
    t0 = time_shift_us + base_us
    st_ev = step_line.events.add()
    st_ev.metadata_id = 1
    st_ev.offset_ps = t0 * _PS_PER_US
    st_ev.duration_ps = dur_us * _PS_PER_US
    _add_int_stat(st_ev, 1, step_num)
    _add_str_stat(st_ev, 2, phase_str)
    _add_int_stat(st_ev, 3, act_tok)
    _add_int_stat(st_ev, 4, elig_tok)

    for meta_id, rel_us, op_dur_us, coll_bytes in (
        (2, 5.0, 10, 0),
        (3, 17.0, 12, _EXPECTED_AR_BYTES),
        (4, 31.5, 15, 0),
        (5, 50.0, 18, _EXPECTED_A2A_BYTES),
    ):
        op_ev = ops_line.events.add()
        op_ev.metadata_id = meta_id
        op_ev.offset_ps = int((t0 + rel_us) * _PS_PER_US)
        op_ev.duration_ps = op_dur_us * _PS_PER_US
        if coll_bytes > 0:
            _add_int_stat(op_ev, 5, _EXPECTED_GROUP_SIZE)
            _add_int_stat(op_ev, 6, coll_bytes)


def _build_synthetic_xspace_bytes(time_shift_us: int = 0) -> bytes:
    """Build a deterministic XSpace proto with device step and XLA op lines."""
    xplane_pb2 = importlib.import_module("vllm_torchtpu.tools.xplane_pb2")
    space = xplane_pb2.XSpace()
    space.hostnames.append("tpu-worker-0")

    meta_plane = space.planes.add()
    meta_plane.id = 99
    meta_plane.name = "/host:metadata"

    cpu_plane = space.planes.add()
    cpu_plane.id = 98
    cpu_plane.name = "/host:CPU"

    dev_plane = space.planes.add()
    dev_plane.id = 1
    dev_plane.name = "/device:TPU:0"
    for ev_id, ev_name in (
        (1, "Step"),
        (2, "qkv_proj"),
        (3, "all-reduce.1"),
        (4, "attn_core"),
        (5, "all-to-all.2"),
    ):
        dev_plane.event_metadata[ev_id].id = ev_id
        dev_plane.event_metadata[ev_id].name = ev_name

    for st_id, st_name in (
        (1, "step_num"),
        (2, "phase"),
        (3, "active_tokens"),
        (4, "fused_eligible_tokens"),
        (5, "group_size"),
        (6, "bytes"),
    ):
        dev_plane.stat_metadata[st_id].id = st_id
        dev_plane.stat_metadata[st_id].name = st_name

    step_line = dev_plane.lines.add()
    step_line.id = 10
    step_line.name = "Steps"
    ops_line = dev_plane.lines.add()
    ops_line.id = 1
    ops_line.name = "XLA Ops"

    w_step = step_line.events.add()
    w_step.metadata_id = 1
    w_step.offset_ps = time_shift_us * _PS_PER_US
    w_step.duration_ps = 140 * _PS_PER_US
    _add_int_stat(w_step, 1, 0)
    _add_str_stat(w_step, 2, "decode")

    for rel_us, meta_id in ((10, 2), (55, 3)):
        w_op = ops_line.events.add()
        w_op.metadata_id = meta_id
        w_op.offset_ps = (time_shift_us + rel_us) * _PS_PER_US
        w_op.duration_ps = 20 * _PS_PER_US

    for spec in (
        (1, 200, 98, "decode", 16, 16),
        (2, 400, 100, "verify", 16, 16),
        (3, 600, 100, "decode", 16, 16),
        (4, 800, 102, "prefill", 16, 0),
    ):
        _append_steady_step_events(step_line, ops_line, spec, time_shift_us)
    return bytes(space.SerializeToString())


def test_calibrate_perfetto_extracts_launch_gap_and_collectives(
    tmp_path: Path,
) -> None:
    """Verify Perfetto trace calibration extracts steady-state gaps and collectives."""
    assert _PERFETTO_FIXTURE.is_file()
    out_json = tmp_path / "perfetto_measurements.json"

    rc = advisor.main(
        [
            "calibrate",
            "--trace",
            str(_PERFETTO_FIXTURE),
            "--chip",
            "tpu7x",
            "--model",
            "Qwen/Qwen3-30B-A3B-FP8",
            "--batch",
            str(_EXPECTED_BATCH),
            "--tp",
            str(_EXPECTED_TP),
            "--out",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "advisor.measurements.v1"
    assert payload["source"]["kind"] == "perfetto"
    assert payload["source"]["batch"] == _EXPECTED_BATCH
    assert payload["source"]["tp"] == _EXPECTED_TP

    lg = payload["launch_gap_us"]
    assert lg["p50"] == pytest.approx(2.5)
    assert lg["p90"] == pytest.approx(3.5)
    assert lg["n"] == _EXPECTED_SINGLE_GAPS

    coll_by_op = {item["op"]: item for item in payload["collectives"]}
    assert set(coll_by_op) == {"all-reduce", "all-to-all"}
    assert coll_by_op["all-reduce"]["group_size"] == _EXPECTED_GROUP_SIZE
    assert coll_by_op["all-reduce"]["bytes"] == _EXPECTED_AR_BYTES
    assert coll_by_op["all-reduce"]["latency_us"] == pytest.approx(12.0)
    assert coll_by_op["all-reduce"]["hop_latency_us"] == pytest.approx(4.0)
    assert coll_by_op["all-reduce"]["n"] == _EXPECTED_STEADY_STEPS
    assert coll_by_op["all-to-all"]["latency_us"] == pytest.approx(18.0)
    assert coll_by_op["all-to-all"]["n"] == _EXPECTED_STEADY_STEPS

    cov = payload["step_coverage"]
    assert cov["fused_eligible_fraction"] == pytest.approx(0.75)
    assert cov["basis"] == "trace"

    st = payload["step_time_us"]
    assert st["p50"] == pytest.approx(100.0)
    assert st["n"] == _EXPECTED_STEADY_STEPS

    profile = advisor.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8")
    chip = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(
        profile,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        chip,
        measurements=out_json,
    )
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["launch_gap_us"]["value"] == pytest.approx(2.5)
    assert report.metrics["collective_latency_us"]["tag"] == "MEASURED"
    assert report.metrics["collective_latency_us"]["value"] == pytest.approx(4.0)
    assert report.metrics["step_coverage"]["tag"] == "MEASURED"
    assert report.metrics["step_coverage"]["value"] == pytest.approx(0.75)


def test_calibrate_xplane_roundtrip_with_in_repo_parser(tmp_path: Path) -> None:
    """Verify single-file and multi-rank XPlane calibration via xplane_pb2."""
    single_pb = tmp_path / "single_rank.xplane.pb"
    single_pb.write_bytes(_build_synthetic_xspace_bytes(time_shift_us=0))

    out_single = tmp_path / "xplane_single_measurements.json"
    payload = advisor.calibrate_trace(
        single_pb,
        out_single,
        chip="tpu7x",
        model="Qwen/Qwen3-30B-A3B-FP8",
    )
    assert out_single.is_file()
    assert payload["schema"] == "advisor.measurements.v1"
    assert payload["source"]["kind"] == "xplane"
    assert payload["launch_gap_us"]["p50"] == pytest.approx(2.5)
    assert payload["launch_gap_us"]["p90"] == pytest.approx(3.5)
    assert payload["launch_gap_us"]["n"] == _EXPECTED_SINGLE_GAPS
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)
    assert payload["step_time_us"]["p50"] == pytest.approx(100.0)

    session_dir = tmp_path / "decode_trace" / "plugins" / "profile" / "2026_11_04"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "rank0_worker0.xplane.pb").write_bytes(
        _build_synthetic_xspace_bytes(time_shift_us=0)
    )
    (session_dir / "rank1_worker1.xplane.pb").write_bytes(
        _build_synthetic_xspace_bytes(time_shift_us=7)
    )

    out_multi = tmp_path / "xplane_multi_measurements.json"
    rc = advisor.main(
        [
            "calibrate",
            "--trace",
            str(tmp_path / "decode_trace"),
            "--out",
            str(out_multi),
        ]
    )
    assert rc == 0
    multi_payload = json.loads(out_multi.read_text(encoding="utf-8"))
    assert multi_payload["launch_gap_us"]["p50"] == pytest.approx(2.5)
    assert multi_payload["launch_gap_us"]["p90"] == pytest.approx(3.5)
    assert multi_payload["launch_gap_us"]["n"] == _EXPECTED_MULTI_GAPS

    profile = advisor.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8")
    chip = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(
        profile,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        chip,
        measurements=out_multi,
    )
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["collective_latency_us"]["tag"] == "MEASURED"


def test_calibrate_float_stats_and_empty_gaps_preserve_assumed_prior(
    tmp_path: Path,
) -> None:
    """Verify float-valued token stats parse cleanly and n=0 gaps stay ASSUMED."""
    trace_file = tmp_path / "single_op_float_stats.json"
    trace_file.write_text(
        json.dumps(
            {
                "traceEvents": [
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
                        "args": {"group_size": 8.0, "bytes": 32768.0},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "empty_gap_measurements.json"
    payload = advisor.calibrate_trace(trace_file, out_json)
    assert payload["launch_gap_us"]["n"] == 0
    assert "p50" not in payload["launch_gap_us"]
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)

    for model_id in ("qwen3.5-397b", "kimi-k3"):
        prof = advisor.resolve_model_profile(model_id)
        rep = advisor.advise(
            prof,
            advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
            advisor.resolve_chip_spec("tpu7x"),
            measurements=out_json,
        )
        assert rep.metrics["launch_gap_us"]["tag"] == "ASSUMED"
        assert rep.metrics["collective_latency_us"]["tag"] == "MEASURED"
        assert rep.metrics["step_coverage"]["tag"] == "MEASURED"


def test_calibrate_rejects_multi_session_and_unannotated_collectives(
    tmp_path: Path,
) -> None:
    """Verify multi-session error guard and collective group_size fallback to tp."""
    phased_root = tmp_path / "phased_run"
    for phase_dir in ("prefill_only", "decode_only"):
        sdir = phased_root / phase_dir / "plugins" / "profile" / "2026_11_04"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "rank0_worker0.xplane.pb").write_bytes(
            _build_synthetic_xspace_bytes(time_shift_us=0)
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

    no_grp_trace = tmp_path / "unannotated_collective.json"
    no_grp_trace.write_text(
        json.dumps(
            {
                "traceEvents": [
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
            }
        ),
        encoding="utf-8",
    )
    skipped_payload = advisor.calibrate_trace(no_grp_trace)
    assert skipped_payload["collectives"] == []

    tp_payload = advisor.calibrate_trace(no_grp_trace, tp=_EXPECTED_FALLBACK_TP)
    assert len(tp_payload["collectives"]) == 1
    assert tp_payload["collectives"][0]["group_size"] == _EXPECTED_FALLBACK_TP
    assert tp_payload["collectives"][0]["hop_latency_us"] == pytest.approx(4.5)


def test_steady_state_steps_drops_compilation_warmup_outlier() -> None:
    """Verifies 2-step trace drops step 0 if it is a severe compilation outlier."""
    from vllm_torchtpu.megakernel.advisor import calibrate
    step0 = calibrate._StepWindow(start_us=0.0, end_us=30000.0, dur_us=30000.0)
    step1 = calibrate._StepWindow(start_us=30000.0, end_us=35000.0, dur_us=5000.0)
    filtered = calibrate._steady_state_steps([step0, step1])
    assert len(filtered) == 1
    assert filtered[0].start_us == 30000.0


def test_steady_state_steps_keeps_normal_two_step_trace() -> None:
    """Verifies 2-step trace preserves step 0 if within normal variation."""
    from vllm_torchtpu.megakernel.advisor import calibrate
    step0 = calibrate._StepWindow(start_us=0.0, end_us=5200.0, dur_us=5200.0)
    step1 = calibrate._StepWindow(start_us=5200.0, end_us=10200.0, dur_us=5000.0)
    filtered = calibrate._steady_state_steps([step0, step1])
    assert len(filtered) == 2
    assert filtered[0].start_us == 0.0
