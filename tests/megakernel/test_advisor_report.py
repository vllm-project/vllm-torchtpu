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
"""Tests for HTML specifications card, flag pills, and whiskers."""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import chips
from vllm_torchtpu.megakernel.advisor import report as advisor_report

if TYPE_CHECKING:
    from pathlib import Path

_WHISKER_RE = re.compile(r'<line\s+class="uncertainty-whisker"\s+([^/>]+)/>')
_ATTR_RE = re.compile(r'([a-zA-Z0-9_-]+)="([^"]*)"')
_MIN_BAR_LABELS = 3


def test_html_renders_provenance_tags_flag_badges_and_onesided_bands() -> None:
    """Verify HTML report renders specifications tags, pills, and whiskers."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_out = advisor.render_advisor_html(sweep)
    body_html = html_out.split("</style>", 1)[1]

    assert "<th>Provenance</th>" not in body_html
    assert "<th>Flags</th>" in body_html
    assert '<span class="prov-tag prov-tag-DERIVED">DERIVED</span>' in body_html
    assert '<span class="prov-tag prov-tag-ASSUMED">ASSUMED</span>' in body_html
    assert '<span class="prov-tag prov-tag-MEASURED">MEASURED</span>' not in body_html

    assert '<span class="flag-pill">' not in body_html
    assert '<span class="kpi-sub">none</span>' in body_html

    multi_call_sweep = advisor.advise_regimes(
        "qwen3.5-397b", "tpu7x", vmem_budget_mib=14.0, op_overhead_us=50.0
    )
    assert "multi_call_unmodeled" in advisor.render_advisor_html(multi_call_sweep)

    unverified_cfg: dict[str, object] = {
        "model_type": "custom_causal_lm_v1",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    unverified_sweep = advisor.advise_regimes(unverified_cfg, "tpu7x")
    unverified_html = advisor.render_advisor_html(unverified_sweep)
    assert "architecture_unverified" in unverified_html

    band_svg = advisor_report.render_step_band_svg(sweep)
    assert band_svg in html_out
    whiskers = _WHISKER_RE.findall(band_svg)
    assert len(whiskers) == len(sweep.regimes)
    for raw_attrs in whiskers:
        attrs = dict(_ATTR_RE.findall(raw_attrs))
        x1 = float(attrs["x1"])
        x2 = float(attrs["x2"])
        floor_us = float(attrs["data-floor"])
        upper_us = float(attrs["data-upper"])
        assert x2 >= x1
        assert upper_us >= floor_us


def test_html_calibrated_provenance_promotes_measured_badge() -> None:
    """Verify calibrated measurements promote MEASURED badge counts in HTML output."""
    cal_data = chips.load_public_calibration("tpu7x")
    assert cal_data is not None
    sweep = advisor.advise_regimes(
        "Qwen/Qwen3-30B-A3B-FP8",
        "tpu7x",
        measurements=cal_data,
    )
    html_out = advisor.render_advisor_html(sweep)
    body_html = html_out.split("</style>", 1)[1]
    assert '<span class="badge badge-measured">CALIBRATED TRACE</span>' in body_html
    assert '<span class="prov-tag prov-tag-MEASURED">MEASURED</span>' in body_html
    assert "xplane:tpu7x_qwen3_30b_a3b_fp8_tp8_ep8" in body_html


def test_html_renders_sourced_facts_and_actionable_columns() -> None:
    """Verify HTML renders top sourced facts ledger and actionable columns."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_out = advisor.render_advisor_html(sweep)
    body_html = html_out.split("</style>", 1)[1]

    assert "Hardware &amp; Model Architecture Specifications" in body_html
    assert "Hardware Specifications" in body_html
    assert "Model Architecture Dimensions" in body_html
    assert '<span class="prov-tag prov-tag-DERIVED">DERIVED</span>' in body_html
    assert '<span class="prov-tag prov-tag-ASSUMED">ASSUMED</span>' in body_html
    assert '<span class="badge badge-assumed">UNCALIBRATED PRIORS</span>' in body_html

    assert '<th class="tabular">Throughput (tok/s/chip)</th>' in body_html
    assert '<th class="tabular">Gain vs per_op</th>' in body_html
    assert "<th>Verdict &amp; Action</th>" in body_html
    assert "theme-toggle" in body_html

    assert "Usable VMEM Ceiling" in body_html
    band_svg = advisor_report.render_step_band_svg(sweep)
    assert "ms</text>" in band_svg

    with pytest.raises(ValueError, match="chip_spec"):
        advisor.render_advisor_html(dataclasses.replace(sweep, chip_spec=None))
    with pytest.raises(ValueError, match="model_profile"):
        advisor.render_advisor_html(dataclasses.replace(sweep, model_profile=None))


def test_every_decision_row_has_collapsed_number_ledger() -> None:
    """Verify each regime row is followed by one ledger with metric and scope tables."""
    for model in ("qwen3.5-397b", "kimi-k3"):
        sweep = advisor.advise_regimes(model, "tpu7x")
        html_out = advisor.render_advisor_html(sweep)
        n_rows = len(sweep.regimes)
        assert html_out.count('<tr class="ledger-row">') == n_rows
        assert html_out.count("<summary>Numbers behind this row:") == n_rows
        assert html_out.count('<th class="tabular">Floor (us)</th>') == n_rows
        for row in sweep.regimes:
            assert {sc["scope"] for sc in row.scopes} == {
                "per_op",
                "single_layer",
                "layer_group",
                "whole_step",
            }


def test_verdict_labels_and_legend_match_what_the_report_shows() -> None:
    """Verify scope-specific verdict labels and a legend without unseen tags."""
    html_out = advisor.render_advisor_html(advisor.advise_regimes("kimi-k3", "tpu7x"))
    assert "FUSION RECOMMENDED" not in html_out
    assert "WHOLE-STEP MEGAKERNEL" in html_out
    assert "LAYER-GROUP MEGAKERNEL" in html_out
    assert "PER-LAYER FUSED KERNEL" in html_out
    # Uncalibrated report: no MEASURED legend pill, only a hint.
    legend = html_out.split('<div class="prov-legend">', 1)[1].split("</div>", 1)[0]
    assert "prov-tag-MEASURED" not in legend
    assert "MEASURED: none yet" in legend
    assert "red dot: per_op floor" in html_out
    assert html_out.count("compute-bound point = ") == len(
        advisor.advise_regimes("kimi-k3", "tpu7x").regimes
    )


def test_decision_table_states_floors_are_not_predictions() -> None:
    """Verify Table 5 tells readers that close calls need an A/B."""
    html_out = advisor.render_advisor_html(advisor.advise_regimes("kimi-k3", "tpu7x"))
    assert "Step bounds are hardware floors, not predictions" in html_out
    assert "A/B the two scopes" in html_out


def test_svg_bar_labels_and_step_band_domain_bounds(tmp_path: Path) -> None:
    """Verify VMEM bar labels track bar tops and step band SVG covers per_op/sub-ms."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_out = advisor.render_advisor_html(sweep)
    bar_label_ys = [
        float(m.group(1))
        for m in re.finditer(
            r'<text x="[0-9.]+" y="([0-9.]+)"[^>]*>[0-9.]+M \([0-9]+ pass',
            html_out,
        )
    ]
    assert len(bar_label_ys) >= _MIN_BAR_LABELS
    # As active token count increases (T=4, 8, 16), stacked bar height grows,
    # so SVG y-coordinates (origin at top) must strictly decrease.
    assert bar_label_ys[0] > bar_label_ys[1] > bar_label_ys[2]

    # Single-regime sweep with 400..600 us step bound and 1500 us per_op floor:
    # 500 us grid line must render as "0.5ms" (not "0ms"), and red per_op dot
    # must sit strictly inside the plot rather than clamping to the right edge.
    entry0 = sweep.regimes[0]
    custom_scopes: tuple[dict[str, object], ...] = tuple(
        {**sc, "step_bound_us": 1500.0 if sc["scope"] == "per_op" else 400.0}
        for sc in entry0.scopes
    )
    fast_entry = dataclasses.replace(
        entry0,
        step_bound_us=400.0,
        step_upper_us=550.0,
        scopes=custom_scopes,
    )
    fast_sweep = dataclasses.replace(sweep, regimes=(fast_entry,))
    band_svg = advisor_report.render_step_band_svg(fast_sweep)
    assert "0.5ms</text>" in band_svg
    assert ">0ms</text>" not in band_svg

    # Empty regimes must render cleanly without ValueError.
    empty_sweep = dataclasses.replace(sweep, regimes=())
    assert "<svg" in advisor_report.render_regime_dot_plot_svg(empty_sweep)
    assert "<svg" in advisor_report.render_step_band_svg(empty_sweep)

    # Terminal table and file writer smoke check.
    table_txt = advisor.format_regime_table(sweep)
    assert "Disaggregated P/D DCN Handoff" in table_txt
    out_file = advisor_report.write_advisor_html(sweep, tmp_path / "advisor.html")
    assert out_file.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


@pytest.mark.parametrize(
    ("verdict", "scope", "expected_tag", "expected_sub"),
    [
        (
            "tie(measure: launch_gap_us)",
            "whole_step",
            '<span class="verdict-tag verdict-tie">TIE</span>',
            '<div class="kpi-sub">measure: <code>launch_gap_us</code></div>',
        ),
        (
            "tie(objective_split: step=TP8xDP1, throughput=TP4xDP2)",
            "whole_step",
            '<span class="verdict-tag verdict-fusion">WHOLE-STEP MEGAKERNEL</span>',
            (
                '<div class="kpi-sub">layout split: '
                "<code>step=TP8xDP1, throughput=TP4xDP2</code></div>"
            ),
        ),
        (
            "tie(objective_split: step=TP8, throughput=TP4)",
            "single_layer",
            '<span class="verdict-tag verdict-layer">PER-LAYER FUSED KERNEL</span>',
            (
                '<div class="kpi-sub">layout split: '
                "<code>step=TP8, throughput=TP4</code></div>"
            ),
        ),
    ],
)
def test_decision_table_verdict_tie_and_objective_split(
    verdict: str,
    scope: str,
    expected_tag: str,
    expected_sub: str,
) -> None:
    """Verify decision table renders distinct badges for ties vs objective splits."""
    profile = advisor.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8")
    chip = advisor.resolve_chip_spec("tpu7x")
    sweep = advisor.advise_regimes(profile, chip)
    assert sweep.regimes

    entry0 = sweep.regimes[0]
    custom_entry = dataclasses.replace(
        entry0,
        verdict=verdict,
        recommended_scope=scope,
    )
    custom_sweep = dataclasses.replace(sweep, regimes=(custom_entry,))
    rows = advisor_report._render_decision_table_rows(custom_sweep)
    assert len(rows) == 2
    assert expected_tag in rows[0]
    assert expected_sub in rows[0]


def test_render_advisor_html_handles_chip_spec_without_sources() -> None:
    """Verify render_advisor_html does not crash when ChipSpec.sources is None."""
    profile = advisor.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8")
    chip_no_sources = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"),
        sources=None,
    )
    sweep = advisor.advise_regimes(profile, chip_no_sources)
    html_out = advisor.render_advisor_html(sweep)
    assert "<!DOCTYPE html>" in html_out or "<html" in html_out
