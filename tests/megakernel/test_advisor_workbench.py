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
import json
import re
from typing import TYPE_CHECKING, cast

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    workbench_engine,
)
from vllm_torchtpu.megakernel.core import errors

if TYPE_CHECKING:
    from pathlib import Path

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
            ),
        )


@pytest.mark.usefixtures("without_declared_megakernels")
@pytest.mark.parametrize("model_name", ["qwen3.5-397b", "kimi-k3", "qwen3.8-2.4t"])
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
            repass_spill_us = cast("float", scope_dict["repass_spill_us"])
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
            step_reconstructed = roof_reconstructed + repass_spill_us + launch_us
            # 4 rounded terms for roof_floor, 6 for step_bound (0.005 us half-ULP each)
            assert roof_reconstructed == pytest.approx(roof_floor, abs=4 * 0.005)
            assert step_reconstructed == pytest.approx(step_bound, abs=6 * 0.005)


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


@pytest.mark.parametrize("model_name", ["qwen3.5-397b", "kimi-k3", "qwen3.8-2.4t"])
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
    assert "ph === 'decode' ? 1" in html_text
    assert "phase === 'decode' ? 1" in html_text
    assert "wbOnHwControl()" in html_text
    assert "wbOnPhaseChange()" in html_text
    assert "wbOnSeqChange()" in html_text
    assert "max-height:164px" in html_text
    assert "wb-knob-pair" in html_text
    assert "overflow-wrap:anywhere" in html_text
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
    ("chip_name", "expected_max_mib"),
    [
        ("tpu7x", "62.0"),
        ("v6e", "126.0"),
    ],
)
def test_workbench_vmem_slider_bounded_by_chip_usable_ceiling(
    chip_name: str,
    expected_max_mib: str,
) -> None:
    """Verify #ctl-vmem range slider min/max reflects each chip's usable VMEM."""
    sweep_report = advisor.advise_regimes("qwen3.5-397b", chip_name)
    html_text = advisor.render_advisor_html(sweep_report)
    vmem_tag = (
        f'<input id="ctl-vmem" type="range" min="2.0" max="{expected_max_mib}" '
        f'step="0.5" value="{expected_max_mib}" oninput="wbOnHwControl()"/>'
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
