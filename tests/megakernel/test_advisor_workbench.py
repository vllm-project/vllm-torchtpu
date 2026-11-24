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
"""Unit tests for megakernel advisor sub-term decomposition and JS engine."""

from __future__ import annotations

import dataclasses
import html
import json
import os
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any, cast

import pytest

from tests.megakernel.test_advisor_calibrate import (
    XLA_MOD_ACTIVE_SPAN_US,
    XLA_MOD_BUSY_US,
    XLA_MOD_FIRST_OP_OFFSET_US,
    XLA_MOD_NUM_LAYERS,
    XLA_MOD_NUM_STEPS,
    XLA_MOD_START_OFFSET_US,
    build_multi_program_xla_modules_perfetto_events,
)
from tests.megakernel.test_advisor_engine_parity import DISPLAY_ROUND_TOL_US
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    models,
    specs,
    workbench_engine,
    workbench_phase,
)
from vllm_torchtpu.megakernel.core import errors

DENYLIST_RE = re.compile(
    r"gmm|ragged_dot|megablox|sparsecore|mxu|tensorcore|vpu|\bllo\b",
    re.IGNORECASE,
)


@pytest.fixture
def without_declared_megakernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip deployment `megakernel` blocks to test core-limit mechanics."""
    for model_key, preset in list(advisor.DEPLOYMENT_PRESETS.items()):
        stripped_sat = (
            dataclasses.replace(preset.saturation, megakernel=None)
            if preset.saturation is not None
            else None
        )
        monkeypatch.setitem(
            advisor.DEPLOYMENT_PRESETS,
            model_key,
            dataclasses.replace(
                preset,
                decode=dataclasses.replace(preset.decode, megakernel=None),
                prefill=dataclasses.replace(preset.prefill, megakernel=None),
                saturation=stripped_sat,
                megakernel_layouts=(),
            ),
        )


@pytest.mark.usefixtures("without_declared_megakernels")
@pytest.mark.parametrize("model_name", ["qwen3.5-397b", "kimi-k3", "glm-5.2"])
def test_scope_estimate_subterms_conserve_step_bound(model_name: str) -> None:
    """Verify ScopeEstimate sub-terms reconstruct roof_floor_us and step_bound_us."""
    sweep_report = advisor.advise_regimes(
        model_name,
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(op_overhead_us=1.5)
        ),
    )
    assert len(sweep_report.regimes) > 0
    for regime in sweep_report.regimes:
        assert regime.workload["phase"] == regime.phase
        for scope_dict in regime.scopes:
            chain_us = cast("float", scope_dict["chain_us"])
            hbm_us = cast("float", scope_dict["hbm_us"])
            comm_us = cast("float", scope_dict["comm_us"])
            fill_drain_us = cast("float", scope_dict["fill_drain_us"])
            launch_us = cast("float", scope_dict["launch_us"])
            roof_floor = cast("float", scope_dict["roof_floor_us"])
            step_bound = cast("float", scope_dict["step_bound_us"])
            is_fused_multi = scope_dict["scope"] in ("whole_step", "layer_group")
            roof_core = (
                max(hbm_us, chain_us + comm_us)
                if is_fused_multi
                else max(chain_us, hbm_us) + comm_us
            )
            roof_reconstructed = roof_core + fill_drain_us
            step_reconstructed = roof_reconstructed + launch_us
            # 4 rounded terms for roof_floor, 5 for step_bound (0.005 us half-ULP each)
            assert roof_reconstructed == pytest.approx(roof_floor, abs=4 * 0.005)
            assert step_reconstructed == pytest.approx(step_bound, abs=5 * 0.005)


def test_workbench_carries_row_limit_and_chain_rule() -> None:
    """Verify the workbench reads a declared row limit and shows the chain rule."""
    sweep_report = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(
                max_rows_per_call=constants.DEFAULT_ROW_LIMIT
            )
        ),
    )
    assert len(sweep_report.regimes) > 0
    for regime in sweep_report.regimes:
        assert regime.workload["max_rows_per_call"] == constants.DEFAULT_ROW_LIMIT

    html_text = advisor.render_advisor_html(sweep_report)
    for required_text in (
        "maxRowsPerCall: rw.max_rows_per_call",
        "t_chain = t_dense + max(t_routed, t_moe)",
        "max(t_hbm, t_chain + t_comm)",
        "max(t_chain, t_hbm) + t_comm",
        "s.roof_floor_us - s.fill_drain_us",
        "Ideal 1-pass",
        "IDEAL TMAX",
    ):
        assert required_text in html_text
    for stale_text in ("t_comp + t_comm", "ICI Collective Overlap"):
        assert stale_text not in html_text


def test_workbench_exposes_zero_snapped_collective_helper() -> None:
    """Verify workbench.js exports exposedCommUs snapping 1-ULP rounding steps."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    for required_text in (
        "function exposedCommUs(s)",
        "if (!(s.comm_us > 0)) return 0;",
        "exposedCommUs,",
    ):
        assert required_text in html_text


def test_workbench_scales_collective_snap_by_calls() -> None:
    """Verify exposedCommUs scales its rounding snap by nCalls and syncs scope."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    for required_text in (
        "const snap = 0.01 * (s.nCalls || 1) + 0.005;",
        "return raw >= snap ? raw : 0;",
        "STATE.selectedTopic.scope = evalA.winner.scope;",
    ):
        assert required_text in html_text


def test_workbench_engine_js_exposes_cleanroom_physics_api() -> None:
    """Verify ES2022 physics evaluator exposes required functions with zero leaks."""
    engine_js = workbench_engine.render_workbench_engine_js()
    assert "window.__MK_ADVISOR_ENGINE" in engine_js
    for symbol_name in (
        "computePerTokenVmem",
        "evaluateWorkload",
        "allReduceCost",
        "expertsHit",
        "devices_per_chip",
    ):
        assert symbol_name in engine_js
    assert DENYLIST_RE.search(engine_js) is None
    body_lines = [
        line
        for line in engine_js.splitlines()
        if "http://www.apache.org/licenses/LICENSE-2.0" not in line
    ]
    body_js = "\n".join(body_lines)
    assert "https://" not in body_js
    assert "http://" not in body_js


@pytest.mark.parametrize(
    ("chip_name", "max_usable_mib"),
    [
        ("tpu7x", 62.0),
        ("v6e", 126.0),
    ],
)
def test_vmem_budget_bounded_by_chip_usable_ceiling(
    chip_name: str,
    max_usable_mib: float,
) -> None:
    """Verify advise() accepts usable ceiling and rejects budgets above chip VMEM."""
    within_budget = advisor.WorkloadSpec(batch=4, vmem_budget_mib=max_usable_mib)
    ok_report = advisor.advise("qwen3.5-397b", within_budget, chip=chip_name)
    assert ok_report.chip == chip_name

    over_budget = advisor.WorkloadSpec(batch=4, vmem_budget_mib=max_usable_mib + 0.5)
    with pytest.raises(errors.MegakernelConfigError, match="exceeds usable VMEM"):
        advisor.advise("qwen3.5-397b", over_budget, chip=chip_name)


def test_custom_chip_spec_preserves_explicit_vmem_usable_bytes(
    tmp_path: Path,
) -> None:
    """Verify --chip-spec JSON preserves explicit vmem_usable_bytes."""
    spec_file = tmp_path / "custom_chip.json"
    spec_file.write_text(
        json.dumps(
            {
                "name": "custom_xpu",
                "devices_per_chip": 2,
                "cores_per_device": 1,
                "vmem_capacity_bytes": 64 * (1 << 20),
                "vmem_usable_bytes": 50 * (1 << 20),
                "hbm_capacity_bytes": 96_000_000_000,
                "hbm_bw_bytes_per_s": 3_600_000_000_000.0,
                "bf16_flops_per_s": 1_150_000_000_000_000.0,
                "ici_egress_bw_bytes_per_s": 300_000_000_000.0,
            }
        ),
        encoding="utf-8",
    )
    resolved = advisor.resolve_chip_spec("custom_xpu", chip_spec_path=spec_file)
    assert resolved.vmem_usable_bytes == 50 * (1 << 20)
    with pytest.raises(errors.MegakernelConfigError, match=r"50\.0 MiB usable"):
        advisor.advise(
            "qwen3.5-397b",
            advisor.WorkloadSpec(batch=4, vmem_budget_mib=50.5),
            chip=resolved,
        )


class _UniqueIdCollector:
    """Collect HTML id attributes from rendered HTML markup."""

    def __init__(self) -> None:
        """Initialize empty id list."""
        self.ids: list[str] = []

    def feed_markup(self, html_text: str) -> list[str]:
        """Extract all id attribute values via regex over static tags."""
        self.ids = re.findall(r'\bid="([^"]+)"', html_text)
        return self.ids


@pytest.mark.parametrize("model_name", ["qwen3.5-397b", "kimi-k3", "glm-5.2"])
def test_workbench_html_three_pane_landmarks_and_unique_ids(model_name: str) -> None:
    """Verify 3-pane Fusion Workbench landmarks, unique DOM IDs, and zero CDN links."""
    sweep_report = advisor.advise_regimes(model_name, "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)

    for required_id in (
        "wb-topbar",
        "view-workbench",
        "wb-left-rail",
        "wb-center-stage",
        "wb-right-inspector",
        "wb-vmem-arena-svg",
        "wb-scope-timeline-svg",
        "wb-waterfall-svg",
        "wb-deriv-drawer",
        "view-audit",
        "btn-swap-ab",
        "btn-vs-perop",
        "btn-reset-all",
        "wb-ab-status",
        "ctl-phase",
        "ctl-batch",
        "ctl-seq-tokens",
        "ctl-context",
        "ctl-vmem",
        "ctl-dma",
        "ctl-alias",
        "ctl-launch-gap",
        "ctl-coll-hop",
        "advisor-data",
    ):
        assert f'id="{required_id}"' in html_text

    collector = _UniqueIdCollector()
    all_ids = collector.feed_markup(html_text)
    assert len(all_ids) == len(set(all_ids))
    assert "wbOnHwControl()" in html_text
    assert "wbOnPhaseChange()" in html_text
    assert "wbOnSeqChange()" in html_text
    assert "wb-knob-pair" in html_text
    assert 'max="1024"' in html_text
    assert 'max="8192"' in html_text
    assert 'step="0.01"' in html_text
    assert 'step="0.001"' in html_text
    assert DENYLIST_RE.search(html_text) is None
    assert "https://" not in html_text
    assert 'href="http://' not in html_text
    assert 'src="http://' not in html_text
    for sel in (
        ".wb-verdict-wrap",
        ".wb-verdict-pill",
        ".wb-verdict-meta",
        ".wb-invalid-banner",
        ".wb-dr-group",
        ".wb-mesh-grid",
        ".wb-eta-details summary",
    ):
        assert sel in html_text


@pytest.mark.parametrize(
    ("chip_name", "expected_max_mib", "expected_init_mib"),
    [
        ("tpu7x", "62.0", "56.0"),
        ("v6e", "126.0", "126.0"),
    ],
)
def test_workbench_vmem_slider_bounded_by_chip_usable_ceiling(
    chip_name: str,
    expected_max_mib: str,
    expected_init_mib: str,
) -> None:
    """Verify #ctl-vmem max reflects usable VMEM and starts at the fit budget."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", chip_name)
    html_text = advisor.render_advisor_html(sweep_report)
    vmem_tag = (
        f'<input id="ctl-vmem" type="range" min="2.0" max="{expected_max_mib}" '
        f'step="0.5" value="{expected_init_mib}" oninput="wbOnHwControl()"/>'
    )
    assert vmem_tag in html_text
    assert 'max="96"' not in html_text


def test_workbench_mesh_dropdowns_contain_only_physical_powers_of_two() -> None:
    """Verify mesh dropdowns exclude non-physical axis sizes such as 3, 6, or 7."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    tp_block = html_text.split('<select id="ctl-tp"', 1)[1].split("</select>", 1)[0]
    options = re.findall(r'value="([^"]+)"', tp_block)
    assert options == ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
    for bad in ("3", "6", "7"):
        assert bad not in options


def test_workbench_verdict_mesh_regime_and_proof_drawer_controls() -> None:
    """Verify verdict bar, feasibility banner, mesh knobs, and proof drawer."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    for snippet in (
        'id="wb-verdict-pill"',
        'id="wb-verdict-summary"',
        'id="wb-invalid-banner"',
        'id="btn-dr-zero"',
        'id="btn-dr-xla"',
        'id="btn-dr-eager"',
        'id="ctl-tp"',
        'id="ctl-ep"',
        'id="ctl-dp"',
        'id="ctl-pp"',
        'id="ctl-dcp"',
        'id="ctl-pcp"',
        'id="ctl-eta-dense"',
        'id="ctl-eta-moe"',
        'id="ctl-eta-hbm"',
        'id="ctl-eta-ici"',
        "TUNE XLA FLAGS FIRST",
        "INVALID CONFIG",
        "Fusion gain (estimate)",
        "expertSlotBytes: 1048576",
        "window.showProof",
        'max="1048576"',
    ):
        assert snippet in html_text
    assert "vmemMip" not in html_text


def test_workbench_phase_map_and_calibration_dropzone_markup() -> None:
    """Verify 2D phase map tab, local FileReader dropzone, and bucket copy UI."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    for required_snippet in (
        'id="btn-stage-phase"',
        'id="wb-stage-phase"',
        'id="wb-phase-map-svg"',
        'id="wb-cal-dropzone"',
        'id="cal-file-input"',
        'id="wb-bucket-schedule-pre"',
        "wbFocusedTopicCard",
        "Selected VMEM:",
        "Waterfall:",
        "Copied!",
        "new FileReader()",
        "new DecompressionStream('gzip')",
        "traceEventsToMeas",
        "STATE.collectiveCalls = parsedCalls",
        "advisor.measurements.v1",
        "advisor.bucket_schedule.v1",
        "advisory_offline_v1",
    ):
        assert required_snippet in html_text
    assert DENYLIST_RE.search(html_text) is None


def test_build_bucket_schedule_and_cli_export(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify build_bucket_schedule coalescing and --bucket-schedule CLI export."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    schedule = advisor.build_bucket_schedule(sweep_report)
    distinct_phases = {reg.phase for reg in sweep_report.regimes}
    assert schedule.schema == "advisor.bucket_schedule.v1"
    assert schedule.consumer_status == "advisory_offline_v1"
    assert schedule.model == "qwen3.5-397b"
    assert len(distinct_phases) <= len(schedule.buckets) <= len(sweep_report.regimes)
    for rule in schedule.buckets:
        assert rule.min_active_tokens <= rule.max_active_tokens
        assert rule.recommended_scope in {
            "whole_step",
            "layer_group",
            "single_layer",
            "per_op",
        }

    out_file = tmp_path / "bucket_schedule.json"
    exit_code = advisor.main(
        ["--model", "qwen3.5-397b", "--bucket-schedule", str(out_file)]
    )
    assert exit_code == 0
    assert out_file.is_file()
    captured = capsys.readouterr()
    assert captured.out == ""

    exit_stdout = advisor.main(["--model", "qwen3.5-397b", "--bucket-schedule", "-"])
    assert exit_stdout == 0
    stdout_json = capsys.readouterr().out
    assert '"schema": "advisor.bucket_schedule.v1"' in stdout_json
    assert '"consumer_status": "advisory_offline_v1"' in stdout_json


def test_phase_map_long_context_oom_hatch_and_conserved_trace_idle(
    tmp_path: Path,
) -> None:
    """Verify 256K/1M phase grid, OOM hatch, and step-idle trace parity."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    assert "262144, 1048576" in html_text
    assert 'id="wb-oom-hatch"' in html_text
    assert "OOM &gt; 80% HBM" in html_text
    assert "us_per_sublayer_boundary" in html_text

    # 4 steps x 400 non-overlapping 2.0 us ops -> 200 us idle;
    # for 60 layers (120 boundaries), p50 == 200 / 120 == 1.6667 us.
    events: list[dict[str, object]] = []
    for step_idx in range(4):
        base_ts = float(step_idx * 1200)
        events.append(
            {
                "name": f"Step {step_idx}",
                "cat": "step_marker",
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": base_ts,
                "dur": 1000.0,
            }
        )
        events.extend(
            [
                {
                    "name": f"fusion.{op_idx}",
                    "cat": "hlo_op",
                    "ph": "X",
                    "pid": 2,
                    "tid": 1,
                    "ts": base_ts + op_idx * 2.5,
                    "dur": 2.0,
                }
                for op_idx in range(400)
            ]
        )
    trace_path = tmp_path / "perfetto_steps.json"
    out_meas = tmp_path / "cal_out.json"
    trace_path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    cal_payload = advisor.calibrate_trace(
        trace_path,
        out_meas,
        options=advisor.CalibrateOptions(chip="tpu7x", num_layers=60),
    )
    py_gap = cal_payload["launch_gap_us"]
    assert "p50" in py_gap
    assert py_gap["p50"] == pytest.approx(200.0 / 120.0, abs=1e-3)
    assert py_gap["step_idle_gap_us"] == pytest.approx(200.0, abs=1e-2)


def _run_workbench_ui_runner(
    tmp_path: Path,
    sweep_report: advisor.AdvisorSweepReport,
    steps: list[dict[str, object]],
    *,
    extra_kpis: bool = False,
    phase: bool = False,
) -> list[dict[str, object]]:
    """Execute advisor_workbench_runner.js via gjs/node and return UI snapshots."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)
    if phase:
        runner_path = Path(__file__).with_name("advisor_workbench_phase_runner.js")
    else:
        runner_path = Path(__file__).with_name("advisor_workbench_runner.js")
        if extra_kpis:
            runner_src = runner_path.read_text(encoding="utf-8").replace(
                '    winnerScope: evA.winner ? evA.winner.scope : "",\n',
                '    winnerScope: evA.winner ? evA.winner.scope : "",\n'
                '    recScope: evA.recScope || "",\n'
                '    kpiScope: elems.get("wb-kpi-scope").textContent,\n'
                '    kpiFloor: elems.get("wb-kpi-floor").textContent,\n'
                '    kpiTmax: elems.get("wb-kpi-tmax").textContent,\n'
                '    kpiTps: elems.get("wb-kpi-tps").textContent,\n'
                '    regimeListHtml: elems.get("wb-regime-list").innerHTML,\n'
                '    abStatus: elems.get("wb-ab-status").innerHTML,\n'
                '    waterfallTitle: elems.get("wb-waterfall-title").textContent,\n',
                1,
            )
            runner_path = tmp_path / "kpi_probe_runner.js"
            runner_path.write_text(runner_src, encoding="utf-8")
    engine_path = (
        Path(workbench_engine.__file__).parent / "assets" / "workbench_engine.js"
    )
    html_text = advisor.render_advisor_html(sweep_report)
    m_seed = re.search(
        r'<script id="advisor-data" type="application/json">(.*?)</script>',
        html_text,
        re.DOTALL,
    )
    assert m_seed is not None
    seed_data = {
        **workbench_engine.build_engine_seed_defaults(),
        **json.loads(m_seed.group(1)),
    }

    in_json = tmp_path / "wb_ui_in.json"
    out_json = tmp_path / "wb_ui_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": seed_data,
                "phase_grid_seed": _phase_grid_seed_from_html(html_text),
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    argv = [js_runtime, str(runner_path), str(in_json), str(out_json)]
    pid = os.posix_spawnp(js_runtime, argv, os.environ)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            if os.waitstatus_to_exitcode(status) != 0:
                msg = "Workbench UI runner exited non-zero"
                raise RuntimeError(msg)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            return cast("list[dict[str, object]]", payload["snapshots"])
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    msg = "Workbench UI runner timed out"
    raise TimeoutError(msg)


def test_workbench_pin_b_preserves_pin_a_and_sliders(tmp_path: Path) -> None:
    """Verify [A]/[B] pins, Swap, vs per_op, arena scope sync, and decode seq lock."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    pinned_row_two = 2
    snaps = _run_workbench_ui_runner(
        tmp_path,
        sweep_report,
        [
            {"op": "set_batch", "value": 64, "tag": "dirty_a"},
            {"op": "pin_regime", "idx": 0, "slot": "B", "tag": "pin_b_same_idx"},
            {
                "op": "pin_regime",
                "idx": pinned_row_two,
                "slot": "B",
                "tag": "pin_b_row2",
            },
            {"op": "swap_ab", "tag": "swapped"},
            {"op": "clear_b", "tag": "cleared_b"},
            {"op": "clear_b", "tag": "cleared_b_again"},
            {"op": "set_arena_scope", "scope": "per_op", "tag": "arena_per_op"},
            {"op": "set_phase", "phase": "verify", "tag": "phase_verify"},
            {"op": "set_phase", "phase": "prefill", "tag": "phase_prefill"},
            {"op": "set_phase", "phase": "decode", "tag": "phase_decode"},
            {"op": "set_seq_tokens", "value": 32, "tag": "seq_drag_on_decode"},
        ],
    )
    by_tag = {str(s["tag"]): s for s in snaps}

    assert by_tag["initial"]["swapDisabled"] is True
    assert by_tag["initial"]["seqDisabled"] is True
    assert by_tag["initial"]["lblSeqTokens"] == "1 (decode)"

    assert by_tag["pin_b_same_idx"]["pinA"] == 0
    assert by_tag["pin_b_same_idx"]["pinB"] == 0
    assert by_tag["pin_b_same_idx"]["customDirty"] is True
    assert by_tag["pin_b_same_idx"]["ctlBatch"] == "64"
    assert by_tag["pin_b_same_idx"]["swapDisabled"] is False

    assert by_tag["pin_b_row2"]["pinA"] == 0
    assert by_tag["pin_b_row2"]["pinB"] == pinned_row_two
    assert by_tag["pin_b_row2"]["ctlBatch"] == "64"

    assert by_tag["swapped"]["pinA"] == pinned_row_two
    assert by_tag["swapped"]["pinB"] == 0

    scope_before = by_tag["swapped"]["ctlArenaScope"]
    assert by_tag["cleared_b"]["pinB"] == -1
    assert by_tag["cleared_b"]["swapDisabled"] is True
    assert by_tag["cleared_b"]["ctlArenaScope"] == scope_before
    assert by_tag["cleared_b_again"]["ctlArenaScope"] == scope_before

    assert by_tag["arena_per_op"]["selectedScope"] == "per_op"
    assert "Mechanism (per_op)" in str(by_tag["arena_per_op"]["derivDrawer"])

    assert by_tag["phase_verify"]["seqDisabled"] is False
    expected_verify_seq_tokens = 4
    expected_prefill_seq_tokens = 256
    assert (
        int(str(by_tag["phase_verify"]["ctlSeqTokens"])) == expected_verify_seq_tokens
    )
    assert (
        int(str(by_tag["phase_prefill"]["ctlSeqTokens"])) == expected_prefill_seq_tokens
    )
    assert by_tag["phase_decode"]["seqDisabled"] is True
    assert by_tag["phase_decode"]["ctlSeqTokens"] == "1"
    assert by_tag["seq_drag_on_decode"]["ctlPhase"] == "decode"
    assert by_tag["seq_drag_on_decode"]["ctlSeqTokens"] == "1"


def test_workbench_dp_phase_map_scales_batch(tmp_path: Path) -> None:
    """Verify 2D Phase Map and live bucket schedule scale batch by DP on DP4xTP2."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    sat_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if int(reg.workload.get("dp", 1)) > 1
    )
    snaps = cast(
        "list[dict[str, object]]",
        _run_workbench_phase_runner(
            tmp_path,
            sweep_report,
            {
                "steps": [
                    {"op": "pin_regime", "idx": sat_idx, "slot": "A", "tag": "pin_dp4"},
                    {
                        "op": "select_phase_cell",
                        "t": 128,
                        "s": 32768,
                        "tag": "cell_128",
                    },
                    {
                        "op": "select_phase_cell",
                        "t": 512,
                        "s": 32768,
                        "tag": "cell_512",
                    },
                ]
            },
        ),
    )
    by_tag = {str(s["tag"]): s for s in snaps}
    expected_tokens_128 = 128
    expected_tokens_512 = 512
    assert by_tag["cell_128"]["activeTokens"] == expected_tokens_128
    assert by_tag["cell_128"]["ctlBatch"] == "512"
    assert by_tag["cell_512"]["activeTokens"] == expected_tokens_512
    assert by_tag["cell_512"]["ctlBatch"] == "2048"

    sched = json.loads(str(by_tag["pin_dp4"]["bucketScheduleJson"]))
    assert sched["buckets"][0]["min_active_tokens"] == 1
    assert sched["buckets"][-1]["max_active_tokens"] == expected_tokens_512


@pytest.mark.usefixtures("without_declared_megakernels")
def test_workbench_inspector_renders_progressive_accordions(tmp_path: Path) -> None:
    """Verify progressive-disclosure <details> dropdowns and LOW ROI topbar badge."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    assert 'id="wb-cuj-hint"' in html_text
    assert 'id="wb-mesh-details"' in html_text
    assert 'id="wb-adv-hw-details"' in html_text
    assert 'onchange="wbOnArenaScopeChange()"' in html_text

    chunked_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if "Chunked Prefill" in reg.label
    )
    snaps = _run_workbench_ui_runner(
        tmp_path,
        sweep_report,
        [{"op": "pin_regime", "idx": chunked_idx, "slot": "A", "tag": "chunked"}],
    )
    by_tag = {str(s["tag"]): s for s in snaps}
    drawer_html = str(by_tag["initial"]["derivDrawer"])
    assert 'class="wb-exec-card"' in drawer_html
    expected_accordion_count = 4
    assert drawer_html.count('class="wb-acc"') == expected_accordion_count
    assert '<details class="wb-acc" open>' in drawer_html
    assert '<details class="wb-topbar-details">' in str(
        by_tag["initial"]["verdictSummary"]
    )
    assert "LOW ROI (&lt;1% PHYSICS)" in str(by_tag["chunked"]["verdictSummary"])

    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    snaps_k3 = _run_workbench_ui_runner(tmp_path, k3_sweep, [], extra_kpis=True)
    init_k3 = snaps_k3[0]
    assert init_k3["winnerScope"] == "whole_step"
    assert init_k3["recScope"] == "per_op"
    assert init_k3["verdictPill"] == "KEEP PER_OP"
    assert str(init_k3["kpiScope"]).startswith("per_op ")
    k3_first = k3_sweep.regimes[0]
    k3_tps = float(str(k3_first.metrics["throughput_tok_s_per_chip"]["value"]))
    assert str(init_k3["kpiFloor"]).startswith(f"{k3_first.step_bound_us:.1f} ~")
    assert init_k3["kpiTps"] == f"{k3_tps:.1f} tok/s/chip"
    assert "T&le;NaN" not in str(init_k3["verdictSummary"])
    assert "Saves <strong>0.0 us" in str(init_k3["verdictSummary"])
    assert "hypothetical: whole_step" in str(init_k3["verdictSummary"])


@pytest.mark.usefixtures("without_declared_megakernels")
def test_workbench_waterfall_card_matches_chain_composition(tmp_path: Path) -> None:
    """Verify waterfall drill-down cards match the chain rule and SVG bar deltas."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    verify_high_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if reg.label.startswith("Spec Verify High")
    )
    steps: list[dict[str, object]] = [
        {"op": "pin_regime", "idx": verify_high_idx, "slot": "A", "tag": "verify_hi"},
    ]
    steps.extend(
        {
            "op": "select_topic",
            "kind": "waterfall",
            "scope": "whole_step",
            "sub": str(row_idx),
            "tag": f"wf_{row_idx}",
        }
        for row_idx in range(4)
    )
    snaps = _run_workbench_ui_runner(tmp_path, sweep_report, steps, phase=True)
    by_tag = {str(s["tag"]): s for s in snaps}

    svg_deltas = [
        float(val)
        for val in re.findall(
            r'class="tabular">([+-]?[0-9.]+) us</text>',
            str(by_tag["verify_hi"]["waterfallSvg"]),
        )
    ]
    waterfall_row_count = 4
    assert len(svg_deltas) == waterfall_row_count

    drawer_core = str(by_tag["wf_0"]["derivDrawer"])
    assert "max(t_chain, t_hbm) overlap floor" in drawer_core
    assert "max(t_comp, t_hbm)" not in drawer_core

    drawer_comm = str(by_tag["wf_1"]["derivDrawer"])
    m_comm_a = re.search(r"\[A\] whole_step: ([+-]?[0-9.]+) us", drawer_comm)
    assert m_comm_a is not None
    assert float(m_comm_a.group(1)) > 0.0

    for row_idx, svg_delta in enumerate(svg_deltas):
        drawer = str(by_tag[f"wf_{row_idx}"]["derivDrawer"])
        m_delta = re.search(r"Delta \(\[A\] - \[B\]\): ([+-]?[0-9.]+) us", drawer)
        assert m_delta is not None
        assert float(m_delta.group(1)) == pytest.approx(svg_delta, abs=1e-9)


def _first_single_layer_verify_batch() -> int:
    """Return the smallest verify batch whose fastest scope is single_layer.

    The scan uses 3 draft tokens on TP8 x EP8 x DCP4 at a 4K context, the
    workload of the spec-verify regimes. The row count where the whole step
    stops winning moves when a priced preset field changes.
    """
    for batch in (8, 16, 32, 64, 128, 256, 512, 1024):
        report = advisor.advise(
            "qwen3.5-397b",
            advisor.WorkloadSpec(
                phase="verify",
                batch=batch,
                spec_tokens=3,
                context=4096,
                tp=8,
                ep=8,
                dcp=4,
            ),
            "tpu7x",
        )
        if report.fastest_scope == "single_layer":
            return batch
    msg = "no verify batch from 8 to 1024 has single_layer as the fastest scope"
    raise AssertionError(msg)


def test_workbench_waterfall_card_tracks_winner_and_zero_sign(tmp_path: Path) -> None:
    """Verify waterfall card tracks evalA.winner, exact zero sign, and 0-comm snap."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    verify_small_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if reg.label.startswith("Spec Verify Small")
    )
    mixed_idx = next(
        idx
        for idx, reg in enumerate(sweep_report.regimes)
        if reg.label.startswith("Mixed Decode+Prefill")
    )
    steps: list[dict[str, object]] = [
        {
            "op": "pin_regime",
            "idx": verify_small_idx,
            "slot": "A",
            "tag": "vsmall",
        },
        {
            "op": "pin_regime",
            "idx": verify_small_idx,
            "slot": "B",
            "tag": "vsmall_b",
        },
        {
            "op": "select_topic",
            "kind": "waterfall",
            "scope": "whole_step",
            "sub": "2",
            "tag": "wf_zero_row",
        },
        {"op": "clear_b", "tag": "clear_b_after_zero"},
        {
            "op": "select_topic",
            "kind": "waterfall",
            "scope": "whole_step",
            "sub": "1",
            "tag": "wf_comm_before_flip",
        },
        {
            "op": "set_batch",
            "value": _first_single_layer_verify_batch(),
            "tag": "flipped_winner",
        },
        {"op": "pin_regime", "idx": mixed_idx, "slot": "A", "tag": "mixed"},
        {"op": "set_batch", "value": 1024, "tag": "mixed_1024"},
        {
            "op": "select_topic",
            "kind": "waterfall",
            "scope": "whole_step",
            "sub": "1",
            "tag": "mixed_comm_card",
        },
    ]
    snaps = _run_workbench_ui_runner(tmp_path, sweep_report, steps, phase=True)
    by_tag = {str(s["tag"]): s for s in snaps}

    zero_svg = re.findall(
        r'class="tabular">([+-]?[0-9.]+ us)</text>',
        str(by_tag["wf_zero_row"]["waterfallSvg"]),
    )
    zero_row_idx = 2
    assert zero_svg[zero_row_idx] == "0.00 us"
    assert "Delta ([A] - [B]): 0.00 us" in str(by_tag["wf_zero_row"]["derivDrawer"])

    assert by_tag["flipped_winner"]["winnerScope"] == "single_layer"
    flip_svg = re.findall(
        r'class="tabular">([+-]?[0-9.]+ us)</text>',
        str(by_tag["flipped_winner"]["waterfallSvg"]),
    )
    flip_drawer = str(by_tag["flipped_winner"]["derivDrawer"])
    assert "[A] single_layer:" in flip_drawer
    assert f"Delta ([A] - [B]): {flip_svg[1]}" in flip_drawer

    mixed_drawer = str(by_tag["mixed_comm_card"]["derivDrawer"])
    assert "[A] whole_step: 0.00 us" in mixed_drawer


def test_workbench_multicall_snap_and_inspecting_scope_sync(tmp_path: Path) -> None:
    """Verify multi-call hidden collectives snap to 0.00 us and scope syncs on flip."""
    row_limit_16 = 16
    batch_32 = 32
    sweep_38 = advisor.advise_regimes(
        "kimi-k3",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(max_rows_per_call=row_limit_16)
        ),
    )
    dec_high_idx = next(
        idx
        for idx, reg in enumerate(sweep_38.regimes)
        if reg.label.startswith("Decode High")
    )
    snaps_mc = _run_workbench_ui_runner(
        tmp_path,
        sweep_38,
        [
            {"op": "pin_regime", "idx": dec_high_idx, "slot": "A", "tag": "dec_hi"},
            {"op": "set_batch", "value": batch_32, "tag": "mc_2calls"},
            {
                "op": "select_topic",
                "kind": "waterfall",
                "scope": "whole_step",
                "sub": "1",
                "tag": "mc_wf_comm",
            },
        ],
        phase=True,
    )
    mc_by_tag = {str(s["tag"]): s for s in snaps_mc}
    dec_high_reg = sweep_38.regimes[dec_high_idx]
    dec_high_par = sweep_38.parallelism
    expected_report = advisor.advise(
        "kimi-k3",
        advisor.WorkloadSpec(
            phase=dec_high_reg.phase,
            batch=batch_32,
            context=dec_high_reg.effective_context,
            tp=dec_high_par.tp,
            ep=dec_high_par.ep,
            dp=dec_high_par.dp,
            pp=dec_high_par.pp,
            dcp=dec_high_par.dcp,
            pcp=dec_high_par.pcp,
            max_rows_per_call=row_limit_16,
        ),
        "tpu7x",
    )
    expected_whole = next(
        scope for scope in expected_report.scopes if scope.scope == "whole_step"
    )
    expected_exposed_comm_us = (
        expected_whole.roof_floor_us
        - expected_whole.fill_drain_us
        - max(expected_whole.chain_us, expected_whole.hbm_us)
    )
    drawer_match = re.search(
        r"\[A\] whole_step: ([+-]?[0-9.]+) us",
        str(mc_by_tag["mc_wf_comm"]["derivDrawer"]),
    )
    assert drawer_match is not None
    assert float(drawer_match.group(1)) == pytest.approx(
        expected_exposed_comm_us, abs=DISPLAY_ROUND_TOL_US
    )

    sweep_35 = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    vsmall_idx = next(
        idx
        for idx, reg in enumerate(sweep_35.regimes)
        if reg.label.startswith("Spec Verify Small")
    )
    snaps_flip = _run_workbench_ui_runner(
        tmp_path,
        sweep_35,
        [
            {"op": "pin_regime", "idx": vsmall_idx, "slot": "A", "tag": "vs"},
            {
                "op": "select_topic",
                "kind": "waterfall",
                "scope": "whole_step",
                "sub": "1",
                "tag": "wf",
            },
            {
                "op": "set_batch",
                "value": _first_single_layer_verify_batch(),
                "tag": "flip",
            },
        ],
        phase=True,
    )
    flip_snap = next(s for s in snaps_flip if s["tag"] == "flip")
    assert flip_snap["selectedScope"] == "single_layer"
    assert flip_snap["ctlArenaScope"] == "single_layer"
    assert "<strong>single_layer (" in str(flip_snap["derivDrawer"])


@pytest.mark.usefixtures("without_declared_megakernels")
def test_live_bucket_schedule_uses_recommended_scope_not_hypothetical(
    tmp_path: Path,
) -> None:
    """Verify buildLiveBucketSchedule uses recScope/recWinner over hypothetical."""
    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    snaps = _run_workbench_ui_runner(tmp_path, k3_sweep, [], phase=True)
    sched = json.loads(str(snaps[0]["bucketScheduleJson"]))
    first_bucket = sched["buckets"][0]
    assert first_bucket["recommended_scope"] == "per_op"
    assert first_bucket["min_active_tokens"] == 1


@pytest.mark.usefixtures("without_declared_megakernels")
def test_workbench_uncalibrated_regime_list_ab_and_waterfall_mark_hypothetical(
    tmp_path: Path,
) -> None:
    """Verify UI regime list, [B] status, KPI mode, and waterfall agree cleanly."""
    k3_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    snaps = _run_workbench_ui_runner(
        tmp_path,
        k3_sweep,
        [{"op": "pin_regime", "idx": 1, "slot": "B", "tag": "pin_b"}],
        extra_kpis=True,
        phase=True,
    )
    init_k3, pin_b = snaps[0], snaps[1]
    expected_mode = k3_sweep.regimes[0].recommended_moe_mode
    assert init_k3["kpiScope"] == f"per_op ({expected_mode})"
    regime_html = str(init_k3["regimeListHtml"])
    k3_bound_str = f"{k3_sweep.regimes[0].step_bound_us:.1f}"
    assert (
        f">per_op</span> &middot; T=4 &middot; {k3_bound_str} us (hyp: whole_step"
        in regime_html
    )
    assert "<strong>[B]</strong>" in str(pin_b["abStatus"])
    assert str(pin_b["abStatus"]).endswith("(per_op)")
    wf_title = str(pin_b["waterfallTitle"])
    assert "[A] whole_step [hypothetical] vs [B]" in wf_title
    assert "[hypothetical])" in wf_title

    q_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    sat_indices = [i for i, r in enumerate(q_sweep.regimes) if "Saturated" in r.label]
    assert [q_sweep.regimes[i].label for i in sat_indices] == [
        "Decode Saturated (B=600 over DP4, T=1, 32K)",
        "Spec Verify Saturated (B=600 over DP4, K=3, 32K)",
    ]
    q_acts = [
        {"op": "pin_regime", "idx": idx, "slot": "A", "tag": f"sat_{idx}"}
        for idx in sat_indices
    ]
    q_snaps = _run_workbench_ui_runner(
        tmp_path, q_sweep, q_acts, extra_kpis=True, phase=True
    )
    first_item_q = str(q_snaps[0]["regimeListHtml"]).split("wbPinRegime(1,")[0]
    assert "(hyp:" not in first_item_q
    assert "[hypothetical]" not in str(q_snaps[0]["waterfallTitle"])
    q_rail = str(q_snaps[0]["regimeListHtml"])
    for pos, idx in enumerate(sat_indices, start=1):
        row_frag = q_rail.split(f"wbPinRegime({idx}, 'A')")[1].split("B</button>")[0]
        assert ">per_op</span>" in row_frag
        assert ">invalid<" not in row_frag
        sat_snap = q_snaps[pos]
        assert sat_snap["recScope"] == "per_op"
        assert sat_snap["verdictPill"] == "KEEP PER_OP"
        assert "(per_op) vs" in str(sat_snap["abStatus"])
        assert "(invalid)" not in str(sat_snap["abStatus"])


def _run_workbench_phase_runner(
    tmp_path: Path,
    sweep_report: advisor.AdvisorSweepReport,
    payload_data: dict[str, Any],
) -> dict[str, Any]:
    """Execute advisor_workbench_phase_runner.js and return phase eval result."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)
    runner_path = Path(__file__).with_name("advisor_workbench_phase_runner.js")
    if not runner_path.is_file():
        msg = f"Missing advisor_workbench_phase_runner.js: {runner_path}"
        raise RuntimeError(msg)

    engine_path = (
        Path(workbench_engine.__file__).parent / "assets" / "workbench_engine.js"
    )
    html_text = advisor.render_advisor_html(sweep_report)
    m_seed = re.search(
        r'<script id="advisor-data" type="application/json">(.*?)</script>',
        html_text,
        re.DOTALL,
    )
    assert m_seed is not None
    seed_data = {
        **workbench_engine.build_engine_seed_defaults(),
        **json.loads(m_seed.group(1)),
    }
    in_json = tmp_path / "wb_phase_in.json"
    out_json = tmp_path / "wb_phase_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": seed_data,
                "phase_grid_seed": _phase_grid_seed_from_html(html_text),
                **payload_data,
            }
        ),
        encoding="utf-8",
    )
    argv = [js_runtime, str(runner_path), str(in_json), str(out_json)]
    pid = os.posix_spawnp(js_runtime, argv, os.environ)
    timeout_s = 60.0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            if os.waitstatus_to_exitcode(status) != 0:
                msg = "Workbench phase runner exited non-zero"
                raise RuntimeError(msg)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and "snapshots" in payload
                and "steps" in payload_data
            ):
                return cast("Any", payload["snapshots"])
            return cast("dict[str, Any]", payload)
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    msg = "Workbench phase runner timed out"
    raise TimeoutError(msg)


def test_js_trace_calibration_parity_and_empty_rejection(tmp_path: Path) -> None:
    """Verify trace normalization, zero-gap omission, and empty cal rejection."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    expected_norm_boundaries = 120
    num_steps = 4
    num_fine_ops = 400
    op_dur_us = 2.0
    op_step_us = 2.5
    step_dur_us = 1000.0
    step_period_us = 1200.0
    expected_step_idle_us = 200.0
    expected_fine_p50_us = expected_step_idle_us / expected_norm_boundaries
    fine_events: list[dict[str, object]] = []
    for step_idx in range(num_steps):
        base_ts = float(step_idx * step_period_us)
        fine_events.append(
            {
                "name": f"Step {step_idx}",
                "cat": "step_marker",
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": base_ts,
                "dur": step_dur_us,
            }
        )
        fine_events.extend(
            [
                {
                    "name": f"fusion.{op_idx}",
                    "cat": "hlo_op",
                    "ph": "X",
                    "pid": 2,
                    "tid": 1,
                    "ts": base_ts + op_idx * op_step_us,
                    "dur": op_dur_us,
                }
                for op_idx in range(num_fine_ops)
            ]
        )
    fine_res = _run_workbench_phase_runner(
        tmp_path,
        sweep_report,
        {
            "action": "trace_to_meas",
            "raw": {"traceEvents": fine_events},
            "fileName": "fine_trace.json",
        },
    )
    fine_meas = cast("dict[str, Any]", fine_res["meas"])
    assert fine_meas["schema"] == "advisor.measurements.v1"
    assert "launch_gap_us" in fine_meas
    assert fine_meas["launch_gap_us"]["p50"] == pytest.approx(
        expected_fine_p50_us, abs=1e-3
    )

    coarse_events = [
        {"name": "op_0", "ph": "X", "pid": 1, "tid": 1, "ts": 100.0, "dur": 50.0},
        {"name": "op_1", "ph": "X", "pid": 1, "tid": 1, "ts": 160.0, "dur": 40.0},
        {"name": "op_2", "ph": "X", "pid": 1, "tid": 1, "ts": 220.0, "dur": 50.0},
    ]
    coarse_res = _run_workbench_phase_runner(
        tmp_path,
        sweep_report,
        {
            "action": "trace_to_meas",
            "raw": {"traceEvents": coarse_events},
            "fileName": "coarse_trace.json",
        },
    )
    coarse_meas = cast("dict[str, Any]", coarse_res["meas"])
    assert "launch_gap_us" not in coarse_meas

    zero_events = [
        {"name": "op_solo", "ph": "X", "pid": 1, "tid": 1, "ts": 100.0, "dur": 50.0}
    ]
    zero_res = _run_workbench_phase_runner(
        tmp_path,
        sweep_report,
        {
            "action": "trace_to_meas",
            "raw": {"traceEvents": zero_events},
            "fileName": "zero_gap.json",
        },
    )
    zero_meas = cast("dict[str, Any]", zero_res["meas"])
    assert "launch_gap_us" not in zero_meas, "0-gap trace must omit launch_gap_us"

    empty_cal = {"schema": "advisor.measurements.v1"}
    empty_res = _run_workbench_phase_runner(
        tmp_path,
        sweep_report,
        {
            "action": "apply_calibration",
            "calPayload": empty_cal,
            "fileName": "empty.json",
        },
    )
    assert empty_res["calibrated"] is False
    assert "Rejected" in str(empty_res["status"])
    expected_launch_gap_str = "0"
    assert empty_res["launchGapValue"] == expected_launch_gap_str

    valid_cal = {
        "schema": "advisor.measurements.v1",
        "launch_gap_us": {"p50": 1.25, "p90": 1.75, "n": 10},
        "step_time_us": {"p50": 420.0, "p90": 450.0, "n": 10},
    }
    valid_res = _run_workbench_phase_runner(
        tmp_path,
        sweep_report,
        {
            "action": "apply_calibration",
            "calPayload": valid_cal,
            "fileName": "valid.json",
        },
    )
    assert valid_res["calibrated"] is True
    expected_compiled_xla = "compiled_xla"
    assert valid_res["dispatchRegime"] == expected_compiled_xla
    expected_cal_gap_str = "1.25"
    assert valid_res["launchGapValue"] == expected_cal_gap_str


def test_coalesced_bucket_schedule_updates_splits(tmp_path: Path) -> None:
    """Verify live bucket schedule updates max_vmem_tokens and splits."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    ref_schedule = advisor.build_bucket_schedule(sweep_report)
    coalesced_ref = [
        b
        for b in ref_schedule.buckets
        if b.phase == "decode" and b.min_active_tokens < b.max_active_tokens
    ]
    assert len(coalesced_ref) > 0, "Reference schedule must contain coalesced rules"

    res = _run_workbench_phase_runner(
        tmp_path,
        sweep_report,
        {
            "action": "bucket_schedule",
            "workload": {
                "model": "qwen3.5-397b",
                "chip": "tpu7x",
                "tp": 8,
                "dp": 1,
                "ep": 8,
                "dcp": 4,
                "context": 4096,
            },
        },
    )
    live_sched = cast("dict[str, Any]", res["schedule"])
    assert live_sched["schema"] == "advisor.bucket_schedule.v1"
    live_buckets = live_sched["buckets"]

    target_ref = coalesced_ref[0]
    matched_bucket = next(
        (
            b
            for b in live_buckets
            if b.get("max_active_tokens") == target_ref.max_active_tokens
            and b.get("min_active_tokens") < b.get("max_active_tokens")
        ),
        None,
    )
    assert matched_bucket is not None, (
        f"Live schedule missing coalesced bucket for max tokens "
        f"{target_ref.max_active_tokens}"
    )
    assert matched_bucket["max_vmem_tokens"] == target_ref.max_vmem_tokens
    assert matched_bucket["num_vmem_splits"] == target_ref.num_vmem_splits


PHASE_MAP_CONTEXT_TOKENS = constants.PHASE_S_GRID[0]
KIMI_DECODE_ROWS = (8, 64, 128)
QWEN_DECODE_ROWS = (16, 64)


def _phase_grid_seed_from_html(html_text: str) -> dict[str, object]:
    """Return the phase map seed that the rendered report hands to the workbench."""
    match = re.search(r"globalThis\.__MK_PHASE_GRID = (\{.*?\});\n", html_text)
    assert match is not None
    return cast("dict[str, object]", json.loads(match.group(1)))


def _phase_map_row(svg_text: str) -> dict[tuple[str, int], dict[str, str]]:
    """Return the map cells of the first context row, keyed by (phase, column)."""
    row: dict[tuple[str, int], dict[str, str]] = {}
    for group in re.findall(r'<g class="wb-clickable" (data-phase=[^>]*)>', svg_text):
        cell = dict(re.findall(r'data-([a-z-]+)="([^"]*)"', group))
        if int(cell["context"]) == PHASE_MAP_CONTEXT_TOKENS:
            row[cell["phase"], int(cell["column"])] = cell
    return row


def _role_step(
    layout: models.ParallelLayout, column: int, *, prefill: bool
) -> advisor.WorkloadSpec:
    """Return the step of one map cell on a deployment role."""
    if prefill:
        # One prompt of `column` tokens in one unchunked, uncached pass.
        return advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=column,
            chunk_size=0,
            prefix_hit_rate=0.0,
            microbatches=0,
            context=PHASE_MAP_CONTEXT_TOKENS,
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
            layer_groups=layout.layer_groups,
        )
    # `column` decode rows per replica.
    return advisor.WorkloadSpec(
        phase="decode",
        batch=column * layout.dp,
        context=PHASE_MAP_CONTEXT_TOKENS,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
    )


def _assert_cells_match_python(
    row: dict[tuple[str, int], dict[str, str]],
    model_name: str,
    columns: tuple[int, ...],
    *,
    prefill: bool,
) -> None:
    """Check map cells against the Python engine on the same deployment role."""
    deployment = models.DEPLOYMENT_PRESETS[model_name]
    layout = deployment.prefill if prefill else deployment.decode
    for column in columns:
        cell = row["prefill" if prefill else "decode", column]
        report = advisor.advise(
            model_name, _role_step(layout, column, prefill=prefill), "tpu7x"
        )
        assert int(cell["active-tokens"]) == report.active_tokens
        assert cell["rec-scope"] == report.recommended_scope
        assert cell["floor-scope"] == report.fastest_scope


def _phase_snapshots(
    tmp_path: Path, model_name: str, steps: list[dict[str, object]]
) -> dict[str, dict[str, object]]:
    """Run the phase workbench of one model and return its snapshots by tag."""
    sweep = advisor.advise_regimes(model_name, "tpu7x")
    snapshots = cast(
        "list[dict[str, object]]",
        _run_workbench_phase_runner(tmp_path, sweep, {"steps": steps}),
    )
    return {str(snapshot["tag"]): snapshot for snapshot in snapshots}


def test_phase_map_shows_recommended_scope_on_each_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the phase map runs each column block on its deployment role."""
    kimi_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    # The rendered report hands the workbench the role layouts of this model.
    report_seed = _phase_grid_seed_from_html(advisor.render_advisor_html(kimi_sweep))
    assert report_seed == workbench_phase.build_phase_grid_seed("kimi-k3")
    assert report_seed["roles"]
    verify_index = next(
        index
        for index, regime in enumerate(kimi_sweep.regimes)
        if regime.label.startswith("Spec Verify Small")
    )
    kimi = _phase_snapshots(
        tmp_path,
        "kimi-k3",
        [
            {"op": "set_phase_map_mode", "mode": "floor", "tag": "floor"},
            {"op": "set_phase_map_mode", "mode": "recommended", "tag": "recommended"},
            {"op": "pin_regime", "idx": verify_index, "slot": "A", "tag": "verify"},
        ],
    )
    prefill_tokens = constants.PHASE_PREFILL_TOKEN_GRID
    kimi_row = _phase_map_row(str(kimi["initial"]["phaseMapSvg"]))
    assert len(kimi_row) == len(constants.PHASE_T_GRID) + len(prefill_tokens)
    for rows in KIMI_DECODE_ROWS:
        assert kimi_row["decode", rows]["scope"] == "whole_step"
    for tokens in prefill_tokens:
        assert kimi_row["prefill", tokens]["scope"] == "per_op"
    _assert_cells_match_python(kimi_row, "kimi-k3", KIMI_DECODE_ROWS, prefill=False)
    _assert_cells_match_python(kimi_row, "kimi-k3", prefill_tokens, prefill=True)
    assert _phase_map_row(str(kimi["recommended"]["phaseMapSvg"])) == kimi_row

    # Floor mode shows the lowest floor of all scopes, which is hypothetical here.
    floor_cell = _phase_map_row(str(kimi["floor"]["phaseMapSvg"]))[
        "prefill", prefill_tokens[0]
    ]
    assert floor_cell["scope"] == floor_cell["floor-scope"]
    assert floor_cell["scope"] != "per_op"
    assert floor_cell["support"] == "hypothetical"

    # A decode cell is plain decode rows whatever regime is pinned.
    verify_row = _phase_map_row(str(kimi["verify"]["phaseMapSvg"]))
    for rows in KIMI_DECODE_ROWS:
        assert int(verify_row["decode", rows]["active-tokens"]) == rows

    qwen_row = _phase_map_row(
        str(_phase_snapshots(tmp_path, "qwen3.5-397b", [])["initial"]["phaseMapSvg"])
    )
    for rows in QWEN_DECODE_ROWS:
        assert qwen_row["decode", rows]["scope"] == "whole_step"
    _assert_cells_match_python(
        qwen_row, "qwen3.5-397b", QWEN_DECODE_ROWS, prefill=False
    )

    # A model without a deployment entry has decode columns only.
    monkeypatch.delitem(models.DEPLOYMENT_PRESETS, "glm-5.2")
    assert "glm-5.2" not in models.DEPLOYMENT_PRESETS
    plain_row = _phase_map_row(
        str(_phase_snapshots(tmp_path, "glm-5.2", [])["initial"]["phaseMapSvg"])
    )
    assert len(plain_row) == len(constants.PHASE_T_GRID)
    assert all(phase == "decode" for phase, _ in plain_row)


def test_js_trace_without_step_event_omits_launch_gap_and_keeps_collectives(
    tmp_path: Path,
) -> None:
    """Verify trace converter omits launch gap and keeps collectives when step-less."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    expected_collective_count = 1
    expected_op = "all-reduce"
    cal_payload: dict[str, object] = {
        "schema": "advisor.measurements.v1",
        "collectives": [
            {
                "op": expected_op,
                "group_size": 8,
                "bytes": 4096,
                "latency_us": 10.0,
                "mean_latency_us": 10.0,
                "n": 2,
            }
        ],
    }
    res = _run_workbench_phase_runner(
        tmp_path,
        sweep_report,
        {
            "action": "trace_to_meas",
            "raw": cal_payload,
            "fileName": "step_less_trace.json",
        },
    )
    meas = cast("dict[str, Any]", res["meas"])
    assert "launch_gap_us" not in meas
    assert len(meas["collectives"]) == expected_collective_count
    assert meas["collectives"][0]["op"] == expected_op


def _eval_workbench_custom_steps(
    tmp_path: Path,
    sweep_report: advisor.AdvisorSweepReport,
    steps: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Run workbench.js custom UI steps in a hermetic temp runner."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)
    base_runner = (
        Path(__file__)
        .with_name("advisor_workbench_runner.js")
        .read_text(encoding="utf-8")
    )
    hook_anchor = (
        '  } else if (op === "select_topic") {\n'
        "    window.wbSelectTopic(step.kind, step.scope, step.sub);\n"
        "  }"
    )
    hook_replacement = (
        '  } else if (op === "select_topic") {\n'
        "    window.wbSelectTopic(step.kind, step.scope, step.sub);\n"
        '  } else if (op === "apply_littles_law") {\n'
        '    doc.getElementById("ctl-out-tps").value = String(step.outTps);\n'
        '    doc.getElementById("ctl-mean-tpot").value = String(step.meanTpot);\n'
        "    window.wbApplyLittlesLaw();\n"
        '  } else if (op === "set_host_heads") {\n'
        '    doc.getElementById("ctl-host-head-per-op").value = String(step.perOpUs);\n'
        '    doc.getElementById("ctl-host-head-fused").value = String(step.fusedUs);\n'
        "    window.wbOnHwControl();\n"
        "  }"
    )
    if hook_anchor not in base_runner:
        msg = "Missing select_topic anchor in advisor_workbench_runner.js"
        raise RuntimeError(msg)
    temp_runner = tmp_path / "custom_wb_runner.js"
    temp_runner.write_text(
        base_runner.replace(hook_anchor, hook_replacement, 1),
        encoding="utf-8",
    )
    engine_path = (
        Path(workbench_engine.__file__).parent / "assets" / "workbench_engine.js"
    )
    html_text = advisor.render_advisor_html(sweep_report)
    m_seed = re.search(
        r'<script id="advisor-data" type="application/json">(.*?)</script>',
        html_text,
        re.DOTALL,
    )
    assert m_seed is not None
    seed_data = {
        **workbench_engine.build_engine_seed_defaults(),
        **json.loads(m_seed.group(1)),
    }
    in_json = tmp_path / "custom_wb_in.json"
    out_json = tmp_path / "custom_wb_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": seed_data,
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    argv = [js_runtime, str(temp_runner), str(in_json), str(out_json)]
    pid = os.posix_spawnp(js_runtime, argv, os.environ)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            if os.waitstatus_to_exitcode(status) != 0:
                msg = "Temp workbench UI runner failed"
                raise RuntimeError(msg)
            snaps = json.loads(out_json.read_text(encoding="utf-8"))["snapshots"]
            return {str(snap["tag"]): cast("dict[str, object]", snap) for snap in snaps}
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    msg = "Temp workbench UI runner timed out"
    raise TimeoutError(msg)


def test_workbench_littles_law_and_host_head_controls(tmp_path: Path) -> None:
    """Verify Decoding Seqs Little's Law helper and host-head equation conservation."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_text = advisor.render_advisor_html(sweep_report)
    assert "Decoding Seqs (B)" in html_text
    assert 'id="ctl-out-tps" type="number" min="0.1" step="1" value=""' in html_text
    assert 'id="ctl-mean-tpot" type="number" min="0.1" step="0.1" value=""' in html_text
    assert 'id="lbl-littles-preview">Enter &gt; 0 tok/s &amp; TPOT<' in html_text
    assert 'id="ctl-host-head-per-op"' in html_text
    assert 'id="ctl-host-head-fused"' in html_text

    output_tok_s = 605.0
    mean_tpot_ms = 13.42
    expected_batch = round(output_tok_s * (mean_tpot_ms / 1000.0))
    snaps = _eval_workbench_custom_steps(
        tmp_path,
        sweep_report,
        [
            {
                "op": "apply_littles_law",
                "outTps": output_tok_s,
                "meanTpot": mean_tpot_ms,
                "tag": "littles_applied",
            },
            {
                "op": "set_host_heads",
                "perOpUs": 1800.0,
                "fusedUs": 350.0,
                "tag": "heads_applied",
            },
        ],
    )
    assert snaps["littles_applied"]["ctlBatch"] == str(expected_batch)
    assert snaps["littles_applied"]["activeTokens"] == expected_batch
    drawer_html = str(snaps["heads_applied"]["derivDrawer"])
    assert " + t_head" in drawer_html
    assert " + 350.00" in drawer_html
    assert "Host Dispatch Head Gain: +1450.00 us" in drawer_html


def _eval_phase_calibration_drawer(
    tmp_path: Path,
    sweep_report: advisor.AdvisorSweepReport,
    cal_payload: dict[str, object],
    file_name: str,
) -> dict[str, object]:
    """Run applyCalibrationObject in a temp phase runner and capture derivDrawer."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)
    base_runner = (
        Path(__file__)
        .with_name("advisor_workbench_phase_runner.js")
        .read_text(encoding="utf-8")
    )
    anchor = (
        '  res.launchGapValue = elems.get("ctl-launch-gap").value;\n'
        "  writeFile(args[1], JSON.stringify(res));"
    )
    replacement = (
        '  res.launchGapValue = elems.get("ctl-launch-gap").value;\n'
        '  res.derivDrawer = elems.get("wb-deriv-drawer").innerHTML;\n'
        "  writeFile(args[1], JSON.stringify(res));"
    )
    if anchor not in base_runner:
        msg = (
            "Missing apply_calibration output anchor in "
            "advisor_workbench_phase_runner.js"
        )
        raise RuntimeError(msg)
    temp_runner = tmp_path / "custom_phase_runner.js"
    temp_runner.write_text(
        base_runner.replace(anchor, replacement, 1),
        encoding="utf-8",
    )
    engine_path = (
        Path(workbench_engine.__file__).parent / "assets" / "workbench_engine.js"
    )
    html_text = advisor.render_advisor_html(sweep_report)
    m_seed = re.search(
        r'<script id="advisor-data" type="application/json">(.*?)</script>',
        html_text,
        re.DOTALL,
    )
    assert m_seed is not None
    seed_data = {
        **workbench_engine.build_engine_seed_defaults(),
        **json.loads(m_seed.group(1)),
    }
    in_json = tmp_path / "custom_phase_cal_in.json"
    out_json = tmp_path / "custom_phase_cal_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": seed_data,
                "phase_grid_seed": _phase_grid_seed_from_html(html_text),
                "action": "apply_calibration",
                "calPayload": cal_payload,
                "fileName": file_name,
            }
        ),
        encoding="utf-8",
    )
    argv = [js_runtime, str(temp_runner), str(in_json), str(out_json)]
    pid = os.posix_spawnp(js_runtime, argv, os.environ)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            if os.waitstatus_to_exitcode(status) != 0:
                msg = "Custom workbench phase runner failed"
                raise RuntimeError(msg)
            return cast(
                "dict[str, object]",
                json.loads(out_json.read_text(encoding="utf-8")),
            )
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    msg = "Custom workbench phase runner timed out"
    raise TimeoutError(msg)


def test_workbench_littles_law_decoding_seqs_and_xla_modules_trace_trim(
    tmp_path: Path,
) -> None:
    """Verify Decoding Seqs Little's Law UI helper and JS XLA Modules trimming."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    snaps = _eval_workbench_custom_steps(
        tmp_path,
        sweep_report,
        [
            {
                "op": "apply_littles_law",
                "outTps": 605.0,
                "meanTpot": 13.42,
                "tag": "littles_8",
            }
        ],
    )
    expected_seqs = round(605.0 * (13.42 / 1000.0))
    assert snaps["littles_8"]["ctlBatch"] == str(expected_seqs)
    assert snaps["littles_8"]["activeTokens"] == expected_seqs

    expected_host_head_us = XLA_MOD_FIRST_OP_OFFSET_US - XLA_MOD_START_OFFSET_US
    expected_step_time_us = XLA_MOD_ACTIVE_SPAN_US
    boundaries = constants.SUBLAYER_BOUNDARIES_PER_LAYER * XLA_MOD_NUM_LAYERS
    expected_launch_gap_us = (XLA_MOD_ACTIVE_SPAN_US - XLA_MOD_BUSY_US) / boundaries

    meas_res = cast(
        "dict[str, object]",
        _run_workbench_phase_runner(
            tmp_path,
            sweep_report,
            {
                "action": "trace_to_meas",
                "raw": build_multi_program_xla_modules_perfetto_events(
                    ranks=(1, 2),
                    host_wraps=True,
                    step0_compile_extra_us=9000.0,
                ),
                "fileName": "xla_mod_multirank.json",
            },
        ),
    )
    js_meas = cast("dict[str, dict[str, float]]", meas_res["meas"])
    expected_multirank_steady = 2 * (XLA_MOD_NUM_STEPS - 1)
    assert js_meas["host_head_us"]["p50"] == pytest.approx(
        expected_host_head_us, abs=1e-2
    )
    assert js_meas["host_head_us"]["n"] == expected_multirank_steady
    assert js_meas["step_time_us"]["p50"] == pytest.approx(
        expected_step_time_us, abs=1e-2
    )
    assert js_meas["step_time_us"]["n"] == expected_multirank_steady
    assert js_meas["launch_gap_us"]["p50"] == pytest.approx(
        expected_launch_gap_us, abs=1e-3
    )
    assert js_meas["launch_gap_us"]["n"] == expected_multirank_steady

    apply_res = _eval_phase_calibration_drawer(
        tmp_path,
        sweep_report,
        {
            "schema": "advisor.measurements.v1",
            "source": {"file": "head_only.json"},
            "host_head_us": {
                "p50": expected_host_head_us,
                "p90": expected_host_head_us + 200.0,
                "n": XLA_MOD_NUM_STEPS - 1,
            },
        },
        "head_only.json",
    )
    assert apply_res["calibrated"] is True
    drawer_after_cal = str(apply_res["derivDrawer"])
    assert "Calibrated Host Head:" in drawer_after_cal
    assert f"{expected_host_head_us:.1f} us" in drawer_after_cal


def _extract_element_inner_html(html_text: str, element_id: str) -> str:
    """Return inner markup of the unique HTML element bearing `element_id`."""
    pattern = rf'<([a-z0-9]+)\b[^>]*\bid="{re.escape(element_id)}"[^>]*>(.*?)</\1>'
    match = re.search(pattern, html_text, re.DOTALL)
    assert match is not None, f"Missing element id={element_id!r}"
    return match.group(2)


def test_workbench_renders_support_tier_and_unmodeled_notes() -> None:
    """Verify workbench topbar renders support tier, sublayer label, and notes."""
    kimi_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    kimi_html = advisor.render_advisor_html(kimi_sweep)
    tier_inner = _extract_element_inner_html(kimi_html, "wb-support-tier")
    assert html.unescape(tier_inner).strip() == kimi_sweep.support_tier
    expected_kimi_title = html.escape(
        specs.SUPPORT_TIER_DESCRIPTIONS[kimi_sweep.support_tier], quote=True
    )
    assert f'id="wb-support-tier" title="{expected_kimi_title}"' in kimi_html

    sublayer_inner = _extract_element_inner_html(kimi_html, "wb-sublayer-label")
    assert (
        html.unescape(sublayer_inner).strip() == kimi_sweep.submodule_labels["decode"]
    )

    assert len(kimi_sweep.unmodeled_notes) > 0
    notes_inner = _extract_element_inner_html(kimi_html, "wb-unmodeled-notes")
    for note in kimi_sweep.unmodeled_notes:
        assert html.escape(note, quote=True) in notes_inner

    glm_sweep = advisor.advise_regimes("glm-5.2", "tpu7x")
    glm_html = advisor.render_advisor_html(glm_sweep)
    glm_tier = _extract_element_inner_html(glm_html, "wb-support-tier")
    assert html.unescape(glm_tier).strip() == glm_sweep.support_tier
    expected_glm_title = html.escape(
        specs.SUPPORT_TIER_DESCRIPTIONS[glm_sweep.support_tier], quote=True
    )
    assert f'id="wb-support-tier" title="{expected_glm_title}"' in glm_html
    assert 'id="wb-unmodeled-notes"' not in glm_html


def test_workbench_auto_regime_upper_band_matches_python(tmp_path: Path) -> None:
    """Verify auto dispatch upper band matches Python on initial load and reset."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    regime0_scopes = sweep_report.regimes[0].scopes
    py_whole_step = next(s for s in regime0_scopes if s["scope"] == "whole_step")
    expected_upper_us = float(str(py_whole_step["step_upper_us"]))
    snaps = _run_workbench_ui_runner(
        tmp_path,
        sweep_report,
        [
            {"op": "dispatch_regime", "regime": "compiled_xla", "tag": "compiled"},
            {"op": "reset_sliders", "tag": "reset"},
        ],
    )
    by_tag = {str(snap["tag"]): snap for snap in snaps}
    # 0.01 us 2-decimal display quantum + 0.005 us float-rounding slack.
    upper_tol_us = 0.015
    initial_upper_us = float(str(by_tag["initial"]["winnerStepUpperUs"]))
    compiled_upper_us = float(str(by_tag["compiled"]["winnerStepUpperUs"]))
    reset_upper_us = float(str(by_tag["reset"]["winnerStepUpperUs"]))
    assert initial_upper_us == pytest.approx(expected_upper_us, abs=upper_tol_us)
    assert compiled_upper_us < initial_upper_us
    assert reset_upper_us == pytest.approx(expected_upper_us, abs=upper_tol_us)
    # Regression check for report.py calibrate CLI hint.
    html_doc = advisor.render_advisor_html(sweep_report)
    assert (
        "advisor calibrate --trace &lt;trace.xplane.pb&gt; --out measurements.json"
        in html_doc
    )


@pytest.mark.parametrize("ops_per_step", [2, 4])
def test_phase_trace_converter_accepts_two_op_explicit_step(
    tmp_path: Path, ops_per_step: int
) -> None:
    """Verify JS phase trace converter normalizes explicit steps at >=2 ops."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    num_layers = 60
    num_steps = 3
    step_span_us = 1000.0
    step_stride_us = 1200.0
    op_dur_us = 235.0
    op_stride_us = 255.0
    events: list[dict[str, object]] = []
    for step_idx in range(num_steps):
        base_ts = float(step_idx) * step_stride_us
        events.append(
            {
                "name": f"Step {step_idx}",
                "cat": "step_marker",
                "ph": "X",
                "pid": 1,
                "tid": 1,
                "ts": base_ts,
                "dur": step_span_us,
            }
        )
        events.extend(
            {
                "name": f"fusion.{op_idx}",
                "cat": "hlo_op",
                "ph": "X",
                "pid": 2,
                "tid": 1,
                "ts": base_ts + float(op_idx) * op_stride_us,
                "dur": op_dur_us,
            }
            for op_idx in range(ops_per_step)
        )
    trace_doc: dict[str, object] = {"traceEvents": events}
    trace_path = tmp_path / f"{ops_per_step}_op_step.json"
    out_meas = tmp_path / f"{ops_per_step}_op_cal.json"
    trace_path.write_text(json.dumps(trace_doc), encoding="utf-8")
    py_cal = advisor.calibrate_trace(
        trace_path,
        out_meas,
        options=advisor.CalibrateOptions(chip="tpu7x", num_layers=num_layers),
    )
    meas_res = cast(
        "dict[str, object]",
        _run_workbench_phase_runner(
            tmp_path,
            sweep_report,
            {
                "action": "trace_to_meas",
                "raw": trace_doc,
                "fileName": f"{ops_per_step}_op_step.json",
            },
        ),
    )
    js_meas = cast("dict[str, dict[str, object]]", meas_res["meas"])
    assert js_meas["launch_gap_us"]["unit"] == "us_per_sublayer_boundary"
    # 5e-5 us half-ULP at QUANTILE_DECIMALS=4 + 5e-5 us float slack.
    gap_tol_us = 1e-4
    assert float(str(js_meas["launch_gap_us"]["p50"])) == pytest.approx(
        float(str(py_cal["launch_gap_us"]["p50"])), abs=gap_tol_us
    )
    # Negative control: a 1-op step below MIN_STEP_OPS_FOR_EXPLICIT_NORM=2
    # must not normalize by 2 * num_layers.
    if ops_per_step == constants.MIN_STEP_OPS_FOR_EXPLICIT_NORM:
        one_op_events = [ev for ev in events if ev["name"] != "fusion.1"]
        one_op_res = cast(
            "dict[str, object]",
            _run_workbench_phase_runner(
                tmp_path,
                sweep_report,
                {
                    "action": "trace_to_meas",
                    "raw": {"traceEvents": one_op_events},
                    "fileName": "one_op_step.json",
                },
            ),
        )
        one_op_meas = cast("dict[str, dict[str, object]]", one_op_res["meas"])
        one_op_launch_gap = one_op_meas.get("launch_gap_us")
        assert (
            one_op_launch_gap is None
            or one_op_launch_gap.get("unit") != "us_per_sublayer_boundary"
        )


def test_workbench_renders_sparse_hc_and_fusion_break_labels() -> None:
    """Verify report.py and workbench JS expose sparse/HC/4-bit/fusion_break labels."""
    eng_table = specs.HostTableBreakSpec(
        name="engram_l1",
        layers=(1,),
        table_bytes_total=101_377_628_352,
        h2d_bytes_per_token=6336.0,
        hbm_resident=False,
        shard_axis="dp_local",
    )
    prof = models.finalize_model_profile(
        models.ModelProfile(
            name="sparse-ui-test",
            num_layers=40,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype="turboquant_4bit_nc",
            sparse_attn=specs.SparseAttnSpec(
                kind="msa",
                window_tokens=128,
                select_tokens=2048,
                index_heads=4,
                index_dim=128,
            ),
            hyper_connection=specs.HyperConnectionSpec(
                kind="mhc", num_streams=4, sinkhorn_iters=20
            ),
            host_tables=(eng_table,),
        )
    )
    sweep = advisor.advise_regimes(
        prof,
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(phase="decode", batch=8, tp=8)
        ),
    )
    html_text = advisor.render_advisor_html(sweep)
    for token in (
        "MSA (Lightning Indexer)",
        "mHC (Sinkhorn manifold)",
        "TurboQuant 4-bit (non-coherent)",
        "Fusion break at host-table or CED boundary",
        "BLOCKER_LABELS",
        "KV_DTYPE_LABELS",
    ):
        assert token in html_text
