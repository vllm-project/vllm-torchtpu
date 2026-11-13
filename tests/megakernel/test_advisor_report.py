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

WHISKER_RE = re.compile(r'<line\s+class="uncertainty-whisker"\s+([^/>]+)/>')
ATTR_RE = re.compile(r'([a-zA-Z0-9_-]+)="([^"]*)"')


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

    pills = set(re.findall(r'<span class="flag-pill">([^<]+)</span>', body_html))
    assert pills == {"above_default_buckets"}
    assert '<span class="kpi-sub">none</span>' in body_html

    # Node 4 dark-theme report.css selectors contract
    assert ".tabular" in html_out
    assert ".prov-tag" in html_out
    assert ".ledger-table" in html_out
    assert ".verdict-tag" in html_out

    if "render_advisor_workbench_html" in advisor.__all__:
        for sel in (
            ".wb-verdict-wrap",
            ".wb-verdict-pill",
            ".wb-verdict-meta",
            ".wb-invalid-banner",
            ".wb-dr-group",
            ".wb-mesh-grid",
            ".wb-eta-details summary",
        ):
            assert sel in html_out

    multi_call_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(
                vmem_budget_mib=14.0, op_overhead_us=50.0
            )
        ),
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
    with pytest.raises(advisor.UnsupportedArchitectureError):
        advisor.advise_regimes(unverified_cfg, "tpu7x")

    unverified_profile = advisor.ModelProfile(
        name="custom_causal_lm_v1",
        num_layers=24,
        hidden_size=2048,
        intermediate_size=5632,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        num_attention_layers=24,
        num_moe_layers=0,
        expert_dtype="fp8",
        model_type="custom_causal_lm_v1",
        architecture_unverified=True,
    )
    unverified_sweep = advisor.advise_regimes(unverified_profile, "tpu7x")
    unverified_html = advisor.render_advisor_html(unverified_sweep)
    assert "architecture_unverified" in unverified_html

    band_svg = advisor_report.render_step_band_svg(sweep)
    assert band_svg in html_out
    whiskers = WHISKER_RE.findall(band_svg)
    assert len(whiskers) == len(sweep.regimes)
    for raw_attrs in whiskers:
        attrs = dict(ATTR_RE.findall(raw_attrs))
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
    assert '<span class="badge badge-assumed">UNCALIBRATED PRIORS</span>' in body_html
    assert '<span class="prov-tag prov-tag-ASSUMED">ASSUMED</span>' in body_html
    assert "architectural_prior:tpu7x_reference_priors" in body_html

    # Passing a real trace measurement promotes to CALIBRATED TRACE / MEASURED.
    trace_data = {
        "schema": "advisor.measurements.v1",
        "source": {
            "kind": "trace",
            "file": "test_trace.json",
            "chip": "tpu7x",
            "phase": "decode",
        },
        "collectives": [
            {
                "op": "all-reduce",
                "group_size": 8,
                "latency_us": 2.5,
                "n": 50,
            }
        ],
        "launch_gap_us": {
            "p50": 1.1,
            "p90": 1.4,
            "n": 50,
        },
    }
    trace_sweep = advisor.advise_regimes(
        "Qwen/Qwen3-30B-A3B-FP8",
        "tpu7x",
        measurements=trace_data,
    )
    trace_html = advisor.render_advisor_html(trace_sweep)
    trace_body = trace_html.split("</style>", 1)[1]
    assert '<span class="badge badge-measured">CALIBRATED TRACE</span>' in trace_body
    assert '<span class="prov-tag prov-tag-MEASURED">MEASURED</span>' in trace_body
    assert "trace:test_trace.json (n=50)" in trace_body


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
        assert html_out.count('<th class="tabular">Estimate (us)</th>') == n_rows
        for row in sweep.regimes:
            assert {scope["scope"] for scope in row.scopes} == {
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
    min_bar_labels = 3
    assert len(bar_label_ys) >= min_bar_labels
    # As active token count increases (T=4, 8, 16), stacked bar height grows,
    # so SVG y-coordinates (origin at top) must strictly decrease.
    assert bar_label_ys[0] > bar_label_ys[1] > bar_label_ys[2]

    # Single-regime sweep with 400..600 us step bound and 1500 us per_op floor:
    # 500 us grid line must render as "0.5ms" (not "0ms"), and red per_op dot
    # must sit strictly inside the plot rather than clamping to the right edge.
    entry0 = sweep.regimes[0]
    custom_scopes: tuple[dict[str, object], ...] = tuple(
        {**scope, "step_bound_us": 1500.0 if scope["scope"] == "per_op" else 400.0}
        for scope in entry0.scopes
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
    html_out = advisor.render_advisor_html(custom_sweep)
    assert expected_tag in html_out
    assert expected_sub in html_out
    assert html_out.count(expected_tag) == 1


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


def test_report_surfaces_gain_split_ring_and_achievable_step() -> None:
    """Verify CLI summary, regime table, and HTML surface gain split and ring slots."""
    report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    summary_txt = advisor.format_advisor_summary(report)
    assert "Fusion gain (estimate):" in summary_txt
    assert "Dispatch Bubble Gain:" in summary_txt
    assert "Derated Achievable Step=" in summary_txt
    assert "ring=(14,10)" in summary_txt

    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    table_txt = advisor.format_regime_table(sweep)
    assert "Fusion gain (estimate)=" in table_txt
    assert "Dispatch Bubble Gain=" in table_txt
    assert "Derated Achievable Step=" in table_txt
    assert "ring=(14,10)" in table_txt

    html_out = advisor.render_advisor_html(sweep)
    assert "physics:" in html_out
    assert "dispatch:" in html_out
    assert "achievable:" in html_out
    assert "ring=<code>(14,10)</code>" in html_out


def test_regime_table_surfaces_pct_headroom_and_low_roi_tag() -> None:
    """Verify regime table surfaces % headroom, LOW ROI tag, and PP prefill."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    by_label = {entry.label: entry for entry in sweep.regimes}
    sat_entry = next(
        entry for label, entry in by_label.items() if "Decode Saturated" in label
    )
    long_prefill = next(
        entry for label, entry in by_label.items() if "Long Prefill" in label
    )
    assert sat_entry.fastest_scope == "whole_step"
    assert sat_entry.recommended_moe_mode == "grouped_on_chip"
    assert long_prefill.recommended_scope == "per_op"
    assert long_prefill.fastest_scope == "single_layer"
    assert long_prefill.recommended_moe_mode == "grouped_on_chip"

    table_txt = advisor.format_regime_table(sweep)
    assert "% vs per_op)" in table_txt
    assert "[LOW ROI <1% PHYSICS]" in table_txt


def test_report_labels_ideal_vmem_ceiling() -> None:
    """Verify reports call the whole_step ceiling an ideal-kernel upper bound."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    html_out = advisor.render_advisor_html(sweep)
    assert "Ideal-Kernel VMEM Ceiling" in html_out
    assert "Full-Step VMEM Ceiling" not in html_out
    assert "not a served limit" in html_out
    assert "ICI Link (all-reduce transfer)" in html_out

    table_txt = advisor.format_regime_table(sweep)
    assert "Ideal-kernel VMEM ceiling (whole_step" in table_txt
    assert "not a served limit" in table_txt
    assert "; ASSUMED, band " in table_txt

    report = advisor.advise(
        advisor.resolve_model_profile("qwen3.5-397b"),
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, dcp=4, ep=8),
        "tpu7x",
    )
    summary_txt = advisor.format_advisor_summary(report)
    assert "us [ASSUMED, band " in summary_txt
    assert "[rows/pass<=" in summary_txt


@pytest.mark.usefixtures("without_declared_megakernels")
def test_charts_and_bucket_schedule_use_recommended_not_hypothetical() -> None:
    """Verify Chart 3 and bucket schedule plot recommended bounds over hypothetical."""
    sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    decode_small = next(r for r in sweep.regimes if r.label.startswith("Decode Small"))
    assert decode_small.recommended_scope == "per_op"
    assert decode_small.hypothetical_best is not None
    per_op_sc = next(s for s in decode_small.scopes if s["scope"] == "per_op")
    per_op_us = float(str(per_op_sc["step_bound_us"]))
    assert decode_small.step_bound_us == pytest.approx(per_op_us)
    assert decode_small.step_bound_us > float(
        str(decode_small.hypothetical_best["step_bound_us"])
    )

    band_svg = advisor_report.render_step_band_svg(sweep)
    whiskers = WHISKER_RE.findall(band_svg)
    decode_attrs = dict(ATTR_RE.findall(whiskers[0]))
    assert float(decode_attrs["data-floor"]) == pytest.approx(per_op_us, abs=0.05)

    html_out = advisor.render_advisor_html(sweep)
    assert "hypothetical:" in html_out

    sched = advisor.build_bucket_schedule(sweep)
    assert sched.schema == advisor.BUCKET_SCHEDULE_SCHEMA
    assert sched.buckets
    decode_rule = next(b for b in sched.buckets if b.phase == "decode")
    assert decode_rule.step_bound_us > 0.0


QWEN35_COMPARISON_WORKLOAD = advisor.WorkloadSpec(
    phase="decode", batch=16, context=4096, objective="both"
)


@pytest.mark.parametrize(
    ("chip", "layout", "expected_chips"),
    [
        # TPU v7x exposes 2 devices per chip, so chips = mesh devices / 2.
        ("tpu7x", advisor.ParallelLayout(tp=8, ep=8, dcp=4), 4),
        ("tpu7x", advisor.ParallelLayout(tp=2, ep=8, dp=4), 4),
        ("tpu7x", advisor.ParallelLayout(tp=8, ep=8, dp=2, dcp=4), 8),
        ("tpu7x", advisor.ParallelLayout(tp=16, ep=16), 8),
        ("tpu7x", advisor.ParallelLayout(tp=1, ep=32, dcp=32), 16),
        ("tpu7x", advisor.ParallelLayout(tp=1, ep=1, pp=8), 4),
        ("tpu7x", advisor.ParallelLayout(tp=8, ep=8, pcp=2), 8),
        # TPU v6e exposes 1 device per chip, so chips = mesh devices.
        ("v6e", advisor.ParallelLayout(tp=8, ep=8), 8),
    ],
)
def test_layout_comparison_rows_print_chips(
    chip: str, layout: advisor.ParallelLayout, expected_chips: int
) -> None:
    """Every layout comparison row carries the physical chip count of its mesh."""
    reference = advisor.ParallelLayout(tp=8, ep=8)
    result = advisor.compare_parallelism_candidates(
        "qwen3.5-397b",
        (reference, layout),
        chip=chip,
        base_workload=QWEN35_COMPARISON_WORKLOAD,
    )
    rows = result["candidates"]
    assert isinstance(rows, list)
    assert [row["layout"] for row in rows] == [reference.label(), layout.label()]
    assert rows[1]["chips"] == expected_chips
    assert all(isinstance(row["chips"], int) and row["chips"] >= 1 for row in rows)


@pytest.mark.parametrize("objective", ["step", "both", "throughput"])
def test_step_ranking_rejects_unequal_chip_layouts(objective: str) -> None:
    """Step time is never ranked across layouts that use different chip counts."""
    base = dataclasses.replace(QWEN35_COMPARISON_WORKLOAD, objective=objective)
    four_chips = advisor.ParallelLayout(tp=8, ep=8)
    eight_chips = advisor.ParallelLayout(tp=16, ep=16)
    unequal = advisor.compare_parallelism_candidates(
        "qwen3.5-397b", (four_chips, eight_chips), chip="tpu7x", base_workload=base
    )
    rows = unequal["candidates"]
    assert isinstance(rows, list)
    assert all(bool(row["hbm_fits"]) and bool(row["valid"]) for row in rows)
    assert unequal["best_step_layout"] == ""
    best_per_chip = max(rows, key=lambda row: row["throughput_tok_s_per_chip"])
    assert unequal["best_throughput_layout"] == best_per_chip["layout"]
    if objective == "throughput":
        assert unequal["verdict"] == unequal["best_throughput_layout"]
    else:
        assert unequal["verdict"] == "not comparable"
    assert [row["chips"] for row in rows] == [4, 8]

    # Equal chip counts keep the step ranking.
    equal = advisor.compare_parallelism_candidates(
        "qwen3.5-397b",
        (
            advisor.ParallelLayout(tp=8, ep=8, dcp=4),
            advisor.ParallelLayout(tp=2, ep=8, dp=4),
        ),
        chip="tpu7x",
        base_workload=base,
    )
    equal_rows = equal["candidates"]
    assert isinstance(equal_rows, list)
    assert all(bool(row["hbm_fits"]) and bool(row["valid"]) for row in equal_rows)
    fastest = min(equal_rows, key=lambda row: row["step_floor_us"])
    assert equal["best_step_layout"] == fastest["layout"]
    assert equal["verdict"] != "not comparable"
