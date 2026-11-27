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
"""Structural, XSS-safety, schedule, CLI, and formatting tests for advisor reports."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
import typing
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from xml.parsers import expat

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import assets
from vllm_torchtpu.megakernel.advisor import report as advisor_report_mod
from vllm_torchtpu.megakernel.advisor import specs as advisor_specs
from vllm_torchtpu.megakernel.core import errors

if TYPE_CHECKING:
    from pathlib import Path

WHISKER_RE = re.compile(r'<line\s+class="uncertainty-whisker"\s+([^/>]+)/>')
ATTR_RE = re.compile(r'([a-zA-Z0-9_-]+)="([^"]*)"')
BAR_LABEL_RE = re.compile(r"^([0-9.]+)M \((\d+) pass")
BAR_LABEL_Y_RE = re.compile(
    r'<text x="[0-9.]+" y="([0-9.]+)"[^>]*>[0-9.]+M \([0-9]+ pass'
)
SVG_BLOCK_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL)
UNESCAPED_ONCLICK_RE = re.compile(r'(?<!&)onclick\s*=\s*"', re.IGNORECASE)
UNIT_AXIS_RE = re.compile(r"\b(?:TP|DP|EP|PP|PCP|DCP)1\b")
MIN_BAR_LABELS = 3
TRACE_STEP_TIME_US = 8200.0
MAX_LEFT_DEAD_SPACE_FRAC = 0.20
MIN_UPPER_GRID_RATIO = 0.40
BADGE_FONT_PX = 11.0
CUSTOM_ROW_CAP = 32
POOL_CAP_FRACTION = 0.45
SOFT_INVALID_TP_DEGREE = 2
SMALL_LATENCY_UPPER_US = 80.0
LARGE_BAND_SPLIT_THRESHOLD = 0.5
LARGE_BAND_THRESHOLD_US = 100_000.0
FLOAT_TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True, slots=True)
class _SvgNode:
    """Parsed SVG element with attributes and accumulated character text."""

    tag: str
    attrib: dict[str, str]
    text: str


def _parse_strict_svg(svg_markup: str) -> list[_SvgNode]:
    """Parse one inline `<svg>` document via strict Expat XML parsing."""
    nodes: list[_SvgNode] = []
    stack: list[tuple[str, dict[str, str], list[str]]] = []

    def _on_start(name: str, attrs: dict[str, str]) -> None:
        """Push an opening XML element onto the parser stack."""
        stack.append(
            (name.rsplit(":", 1)[-1], {k.lower(): v for k, v in attrs.items()}, [])
        )

    def _on_data(chunk: str) -> None:
        """Append character data to the active XML element."""
        if stack:
            stack[-1][2].append(chunk)

    def _on_end(_name: str) -> None:
        """Pop a closed XML element and record its `_SvgNode`."""
        tag, attrib, chunks = stack.pop()
        nodes.append(_SvgNode(tag=tag, attrib=attrib, text="".join(chunks)))

    parser = expat.ParserCreate()
    parser.StartElementHandler = _on_start
    parser.CharacterDataHandler = _on_data
    parser.EndElementHandler = _on_end
    is_final = True
    parser.Parse(svg_markup, is_final)
    assert any(node.tag == "svg" for node in nodes)
    svg_root = next(node for node in nodes if node.tag == "svg")
    return [svg_root, *(node for node in nodes if node is not svg_root)]


def _assert_zero_js_and_valid_svgs(
    html_doc: str, css_text: str
) -> list[list[_SvgNode]]:
    """Assert DOCTYPE, embedded CSS, zero JS, and non-clipped XML-valid SVGs."""
    assert html_doc.startswith("<!DOCTYPE html>")
    assert 'data-theme="dark"' not in html_doc
    assert css_text in html_doc
    assert "<script" not in html_doc.lower()
    assert "onclick=" not in html_doc.lower()
    assert "localStorage" not in html_doc
    svg_blocks = SVG_BLOCK_RE.findall(html_doc)
    assert svg_blocks
    parsed_svgs: list[list[_SvgNode]] = []
    for block in svg_blocks:
        assert 'viewBox="' in block
        nodes = _parse_strict_svg(block)
        parsed_svgs.append(nodes)
        root_node = next(node for node in nodes if node.tag == "svg")
        view_box = [float(part) for part in root_node.attrib["viewbox"].split()]
        svg_width, svg_height = view_box[2], view_box[3]
        for text_elem in nodes:
            if text_elem.tag != "text":
                continue
            text_content = text_elem.text
            x_pos = float(text_elem.attrib["x"])
            y_pos = float(text_elem.attrib["y"])
            font_px = float(text_elem.attrib.get("font-size", "13"))
            anchor = text_elem.attrib.get("text-anchor", "start")
            width_px = advisor_report_mod.svg_text_width(text_content, font_px)
            if anchor == "middle":
                left_x, right_x = x_pos - width_px / 2.0, x_pos + width_px / 2.0
            elif anchor == "end":
                left_x, right_x = x_pos - width_px, x_pos
            else:
                left_x, right_x = x_pos, x_pos + width_px
            assert left_x >= -FLOAT_TOLERANCE, (
                f"Clipped left SVG text {text_content!r}: {left_x}"
            )
            assert right_x <= svg_width + FLOAT_TOLERANCE, (
                f"Clipped right SVG text {text_content!r}: {right_x} > {svg_width}"
            )
            assert 0.0 <= y_pos <= svg_height
    return parsed_svgs


def test_html_escaping_and_xss_safety() -> None:
    """Verify untrusted strings across all report fields are HTML/SVG-escaped."""
    xss_payload = '<script>alert("xss")</script>"><img src=x onerror=alert(1)>'
    base_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    tainted_metric = typing.cast(
        "advisor_specs.ProvenanceMetric",
        {
            "value": xss_payload,
            "unit": xss_payload,
            "tag": xss_payload,
            "source": xss_payload,
            "band": [0.0, 1.0],
        },
    )
    tainted_scope = dataclasses.replace(
        base_report.scopes[0],
        reason=xss_payload,
        binding_unit=xss_payload,
    )
    tainted_report = dataclasses.replace(
        base_report,
        model=xss_payload,
        chip=xss_payload,
        rationale=xss_payload,
        recommended_moe_mode=xss_payload,
        verdict=f"invalid_config({xss_payload})",
        confidence=xss_payload,
        coverage=xss_payload,
        ranking_metric=xss_payload,
        binding_unit=xss_payload,
        core_support=typing.cast("advisor_specs.SupportStatus", xss_payload),
        core_blockers=(xss_payload,),
        core_ready_scope=typing.cast("advisor.FusionScope", xss_payload),
        flags=(xss_payload,),
        scopes=(tainted_scope, *base_report.scopes[1:]),
        metrics={**base_report.metrics, "custom_metric": tainted_metric},
        unresolved_assumptions=(
            {
                "name": xss_payload,
                "value": xss_payload,
                "band": [0, 1],
                "unit": xss_payload,
                "affects": xss_payload,
                "how_to_settle": xss_payload,
            },
        ),
    )
    single_html = advisor.render_advisor_html(tainted_report)
    assert "<script" not in single_html.lower()
    assert "<img" not in single_html.lower()
    assert UNESCAPED_ONCLICK_RE.search(single_html) is None
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in single_html

    base_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    tainted_entry = dataclasses.replace(
        base_sweep.regimes[0],
        label=xss_payload,
        layout=xss_payload,
        rationale=xss_payload,
        recommended_moe_mode=xss_payload,
        core_support=typing.cast("advisor_specs.SupportStatus", xss_payload),
        core_blockers=(xss_payload,),
        core_ready_scope=typing.cast("advisor.FusionScope", xss_payload),
        evidence=typing.cast("advisor_specs.SupportStatus", xss_payload),
        blockers=(xss_payload,),
        flags=(xss_payload,),
    )
    tainted_search = (
        dataclasses.replace(
            base_sweep.layer_group_search, note=xss_payload, layout=xss_payload
        )
        if base_sweep.layer_group_search is not None
        else None
    )
    tainted_handoff = (
        dataclasses.replace(
            base_sweep.disagg_handoff,
            dcn_provenance=typing.cast(
                'typing.Literal["ASSUMED", "MEASURED"]', xss_payload
            ),
        )
        if base_sweep.disagg_handoff is not None
        else None
    )
    tainted_profile = (
        dataclasses.replace(base_sweep.model_profile, model_type=xss_payload)
        if base_sweep.model_profile is not None
        else None
    )
    tainted_sweep = dataclasses.replace(
        base_sweep,
        model=xss_payload,
        chip=xss_payload,
        parallelism=dataclasses.replace(base_sweep.parallelism, topology=xss_payload),
        chip_sources={"vmem_usable_bytes": xss_payload},
        model_sources={"config": xss_payload},
        regimes=(tainted_entry,),
        tailoring_guide=(xss_payload,),
        layer_group_search=tainted_search,
        disagg_handoff=tainted_handoff,
        model_profile=tainted_profile,
        submodule_labels={"decode": xss_payload},
    )
    sweep_html = advisor.render_advisor_html(tainted_sweep)
    assert "<script" not in sweep_html.lower()
    assert "<img" not in sweep_html.lower()
    assert UNESCAPED_ONCLICK_RE.search(sweep_html) is None
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in sweep_html


def test_static_html_and_svg_structure_single_and_sweep(tmp_path: Path) -> None:
    """Verify static HTML/SVG structure, XML validity, and empty-sweep safety."""
    css_text = assets.load_report_css()
    assert ".chart-svg" in css_text
    assert '[data-theme="auto"]' not in css_text
    assert not hasattr(assets, "load_asset")

    single_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    single_html = advisor.render_advisor_html(single_report)
    _assert_zero_js_and_valid_svgs(single_html, css_text)
    assert len(WHISKER_RE.findall(single_html)) == len(single_report.scopes)
    body_without_style = re.sub(
        r"<style\b[^>]*>.*?</style>", "", single_html, flags=re.DOTALL | re.IGNORECASE
    )
    assert css_text not in body_without_style
    for scope_name in ("per_op", "single_layer", "layer_group", "whole_step"):
        assert scope_name in body_without_style

    empty_scope_html = advisor.render_advisor_html(
        dataclasses.replace(single_report, scopes=())
    )
    _assert_zero_js_and_valid_svgs(empty_scope_html, css_text)

    short_scope = dataclasses.replace(
        single_report.scopes[0],
        step_floor_strict_us=40.0,
        step_bound_us=45.0,
        step_upper_us=SMALL_LATENCY_UPPER_US,
        step_achievable_us=60.0,
    )
    short_latency_html = advisor.render_advisor_html(
        dataclasses.replace(single_report, scopes=(short_scope,))
    )
    _assert_zero_js_and_valid_svgs(short_latency_html, css_text)

    single_file = advisor.write_advisor_html(single_report, tmp_path / "single.html")
    assert single_file.read_text(encoding="utf-8") == single_html

    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    sweep_html = advisor.render_advisor_html(sweep)
    _assert_zero_js_and_valid_svgs(sweep_html, css_text)
    assert sweep_html.count('<tr class="ledger-row">') == len(sweep.regimes)
    assert sweep_html.count('<details class="prov-details">') == len(sweep.regimes)

    bar_ys = [float(match.group(1)) for match in BAR_LABEL_Y_RE.finditer(sweep_html)]
    assert len(bar_ys) >= MIN_BAR_LABELS
    assert bar_ys[0] > bar_ys[1] > bar_ys[2]

    empty_sweep = dataclasses.replace(sweep, regimes=())
    empty_html = advisor.render_advisor_html(empty_sweep)
    _assert_zero_js_and_valid_svgs(empty_html, css_text)
    assert "No regime rows evaluated" in empty_html

    no_sources_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"), sources=None
    )
    no_sources_sweep = advisor.advise_regimes("qwen3.5-397b", no_sources_chip)
    assert advisor.render_advisor_html(no_sources_sweep).startswith("<!DOCTYPE html>")

    sweep_file = advisor.write_advisor_html(sweep, tmp_path / "sub" / "sweep.html")
    assert sweep_file.read_text(encoding="utf-8") == sweep_html

    with pytest.raises(ValueError, match="chip_spec"):
        advisor.render_advisor_html(dataclasses.replace(sweep, chip_spec=None))
    with pytest.raises(ValueError, match="model_profile"):
        advisor.render_advisor_html(dataclasses.replace(sweep, model_profile=None))
    with pytest.raises(TypeError, match="Expected AdvisorReport or AdvisorSweepReport"):
        advisor.render_advisor_html(json.loads('"invalid"'))


@pytest.mark.parametrize("model_slug", list(advisor.MODEL_PRESETS))
def test_vmem_chart_single_pass_bars_never_exceed_budget_line(model_slug: str) -> None:
    """Verify Chart 1 adaptive ring sizing keeps all 1-pass bars <= budget."""
    css_text = assets.load_report_css()
    for headroom in (False, True):
        sweep = advisor.advise_regimes(
            model_slug,
            "tpu7x",
            options=advisor.RegimeSweepOptions(
                base_workload=advisor.WorkloadSpec(vmem_fit_headroom=headroom)
            ),
        )
        html_out = advisor.render_advisor_html(sweep)
        roots = _assert_zero_js_and_valid_svgs(html_out, css_text)
        vmem_nodes = roots[0]
        vmem_text = " ".join(node.text for node in vmem_nodes if node.text)
        assert "2-Slot" not in vmem_text
        assert "Adaptive Weight Ring" in vmem_text

        checked_bars = 0
        cap_text = ""
        for elem in vmem_nodes:
            if elem.tag == "text" and elem.text:
                if elem.text.startswith("Usable VMEM Ceiling:"):
                    cap_text = elem.text
                if (match := BAR_LABEL_RE.match(elem.text)) is not None:
                    plotted_mib = float(match.group(1))
                    passes = int(match.group(2))
                    checked_bars += 1
                    if passes == 1:
                        assert plotted_mib <= sweep.vmem_budget_mib + FLOAT_TOLERANCE
        assert checked_bars == len(advisor_report_mod.VMEM_TOKEN_TICKS)
        assert cap_text
        cap_rects = [
            float(elem.attrib["width"])
            for elem in vmem_nodes
            if elem.tag == "rect" and elem.attrib.get("stroke") == "#EF4444"
        ]
        assert cap_rects
        assert cap_rects[0] >= advisor_report_mod.svg_text_width(
            cap_text, BADGE_FONT_PX
        )

    capped_sweep = advisor.advise_regimes(
        model_slug,
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(max_rows_per_call=CUSTOM_ROW_CAP)
        ),
    )
    capped_html = advisor.render_advisor_html(capped_sweep)
    capped_roots = _assert_zero_js_and_valid_svgs(capped_html, css_text)
    assert "row cap" in " ".join(node.text for node in capped_roots[0] if node.text)


@pytest.mark.parametrize("model_slug", list(advisor.MODEL_PRESETS))
def test_latency_band_svg_dead_space_and_grid_coverage(model_slug: str) -> None:
    """Verify dynamic log10 domain bounds and upper gridlines in band charts."""
    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    band_svg = advisor_report_mod.render_step_band_svg(sweep)
    nodes = _parse_strict_svg(band_svg)

    whiskers = [
        elem
        for elem in nodes
        if elem.tag == "line" and elem.attrib.get("class") == "uncertainty-whisker"
    ]
    assert len(whiskers) == len(sweep.regimes)
    for whisker in whiskers:
        assert float(whisker.attrib["x2"]) >= float(whisker.attrib["x1"])
        assert float(whisker.attrib["data-upper"]) >= float(
            whisker.attrib["data-floor"]
        )

    row_lines = [
        elem
        for elem in nodes
        if elem.tag == "line" and elem.attrib.get("stroke-dasharray") == "2,4"
    ]
    assert row_lines
    x_start = float(row_lines[0].attrib["x1"])
    x_end = float(row_lines[0].attrib["x2"])
    plot_width = x_end - x_start
    min_whisker_x1 = min(float(w.attrib["x1"]) for w in whiskers)
    assert (min_whisker_x1 - x_start) / plot_width <= MAX_LEFT_DEAD_SPACE_FRAC

    grid_ticks_us: list[float] = []
    for elem in nodes:
        if (
            elem.tag != "text"
            or not elem.text
            or elem.attrib.get("text-anchor") != "middle"
        ):
            continue
        label = elem.text.strip()
        if label.endswith("ms"):
            grid_ticks_us.append(float(label[:-2]) * 1000.0)
        elif label.endswith("us"):
            grid_ticks_us.append(float(label[:-2]))
    assert grid_ticks_us
    max_upper_us = max(float(w.attrib["data-upper"]) for w in whiskers)
    expected_ratio = (
        LARGE_BAND_SPLIT_THRESHOLD
        if max_upper_us > LARGE_BAND_THRESHOLD_US
        else MIN_UPPER_GRID_RATIO
    )
    assert max(grid_ticks_us) >= expected_ratio * max_upper_us


def test_static_html_surfaces_calibrated_provenance_and_layer_groups() -> None:
    """Verify calibrated and uncalibrated trace provenance and calibration hint."""
    uncalibrated_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    uncalibrated_html = advisor.render_advisor_html(uncalibrated_sweep)
    expected_cmd = (
        "python -m vllm_torchtpu.megakernel.advisor calibrate "
        "--trace &lt;trace.xplane.pb&gt; --model kimi-k3 --out measurements.json"
    )
    assert expected_cmd in uncalibrated_html
    assert (
        "run python -m vllm_torchtpu.megakernel.advisor calibrate" in uncalibrated_html
    )

    trace_payload = {
        "schema": "advisor.measurements.v1",
        "source": {"kind": "xplane", "file": "calibrated.xplane.pb"},
        "launch_gap_us": {"p50": 1.2, "p90": 1.5, "n": 40},
        "step_time_us": {"p50": TRACE_STEP_TIME_US, "n": 40},
    }
    sweep = advisor.advise_regimes("kimi-k3", "tpu7x", measurements=trace_payload)
    html_out = advisor.render_advisor_html(sweep)
    assert "CALIBRATED TRACE" in html_out
    assert "calibrated.xplane.pb" in html_out
    assert "Layer-Group Partition Search" in html_out
    assert "outside_core_limits" in html_out
    assert "core_ready_scope=<code>per_op</code>" in html_out
    assert "kv_or_state_dtype_not_in_core" in html_out


@pytest.mark.parametrize("model_slug", list(advisor.MODEL_PRESETS))
def test_bucket_schedule_invariants_and_json_roundtrip(
    model_slug: str, tmp_path: Path
) -> None:
    """Verify bucket exclusion of OOM/invalid rows, coalescing, types, and JSON."""
    hints = typing.get_type_hints(advisor.BucketScheduleEntry)
    assert hints["phase"] is advisor.Phase
    assert hints["recommended_scope"] is advisor.FusionScope
    assert hints["core_support"] is advisor_specs.SupportStatus
    assert hints["core_ready_scope"] is advisor.FusionScope

    sweep = advisor.advise_regimes(model_slug, "tpu7x")
    schedule = advisor.build_bucket_schedule(sweep)
    valid_entries = [
        entry
        for entry in sweep.regimes
        if entry.hbm_fits and not entry.verdict.startswith("invalid_config(")
    ]
    valid_phases = {entry.phase for entry in valid_entries}
    assert schedule.schema == advisor.BUCKET_SCHEDULE_SCHEMA
    assert schedule.consumer_status == advisor.BUCKET_CONSUMER_STATUS
    assert schedule.model == model_slug
    assert schedule.default_layout == sweep.parallelism.label()
    assert len(valid_phases) <= len(schedule.buckets) <= len(valid_entries)

    oom_tuples = {
        (entry.phase, entry.layout, entry.active_tokens, entry.effective_context)
        for entry in sweep.regimes
        if not entry.hbm_fits
    }
    if oom_tuples:
        for bucket in schedule.buckets:
            for tok in (bucket.min_active_tokens, bucket.max_active_tokens):
                assert (
                    bucket.phase,
                    bucket.layout,
                    tok,
                    bucket.effective_context,
                ) not in oom_tuples

    prev_by_group: dict[tuple[str, str], advisor.BucketScheduleEntry] = {}
    for bucket in schedule.buckets:
        assert not bucket.verdict.startswith("invalid_config(")
        assert bucket.min_active_tokens <= bucket.max_active_tokens
        assert bucket.step_bound_us > 0.0
        assert bucket.max_vmem_tokens > 0
        assert bucket.num_vmem_splits >= 1
        if bucket.recommended_scope == "per_op":
            assert bucket.num_vmem_splits == 1
        elif bucket.max_active_tokens > bucket.max_vmem_tokens:
            expected_min_splits = math.ceil(
                bucket.max_active_tokens / bucket.max_vmem_tokens
            )
            assert bucket.num_vmem_splits >= expected_min_splits
        group_key = (bucket.phase, bucket.layout)
        prev_bucket = prev_by_group.get(group_key)
        if prev_bucket is not None:
            if prev_bucket.max_active_tokens == bucket.min_active_tokens:
                assert prev_bucket.effective_context != bucket.effective_context
            else:
                assert prev_bucket.max_active_tokens < bucket.min_active_tokens
            assert (
                prev_bucket.recommended_scope,
                prev_bucket.recommended_moe_mode,
                prev_bucket.verdict,
                prev_bucket.core_support,
                prev_bucket.core_ready_scope,
                prev_bucket.core_blockers,
            ) != (
                bucket.recommended_scope,
                bucket.recommended_moe_mode,
                bucket.verdict,
                bucket.core_support,
                bucket.core_ready_scope,
                bucket.core_blockers,
            )
        prev_by_group[group_key] = bucket

    out_path = tmp_path / "nested" / "bucket_schedule.json"
    written_path = advisor.write_bucket_schedule(sweep, out_path)
    assert written_path == out_path
    roundtrip = json.loads(out_path.read_text(encoding="utf-8"))
    assert roundtrip == schedule.to_dict()


def test_bucket_schedule_excludes_all_oom_and_preserves_verdicts() -> None:
    """Verify all-OOM/invalid sweeps emit empty buckets and verdicts stay split."""
    all_oom_sweep = advisor.advise_regimes(
        "kimi-k3",
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=8, ep=8),
    )
    assert all(not entry.hbm_fits for entry in all_oom_sweep.regimes)
    assert advisor.build_bucket_schedule(all_oom_sweep).buckets == ()

    invalid_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=SOFT_INVALID_TP_DEGREE, ep=1),
    )
    assert all(
        entry.verdict.startswith("invalid_config(") for entry in invalid_sweep.regimes
    )
    assert advisor.build_bucket_schedule(invalid_sweep).buckets == ()

    base_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    base_entry = next(entry for entry in base_sweep.regimes if entry.hbm_fits)
    tie_entry = dataclasses.replace(
        base_entry,
        active_tokens=8,
        effective_context=4096,
        max_vmem_tokens=400,
        verdict="tie(measure:launch_gap_us)",
    )
    definite_entry = dataclasses.replace(
        base_entry,
        active_tokens=16,
        effective_context=4096,
        max_vmem_tokens=200,
        verdict="megakernel_likely",
    )
    definite_merge = dataclasses.replace(
        base_entry,
        active_tokens=32,
        effective_context=8192,
        max_vmem_tokens=120,
        verdict="megakernel_likely",
    )
    split_sweep = dataclasses.replace(
        base_sweep, regimes=(tie_entry, definite_entry, definite_merge)
    )
    split_schedule = advisor.build_bucket_schedule(split_sweep)
    assert [bucket.verdict for bucket in split_schedule.buckets] == [
        "tie(measure:launch_gap_us)",
        "megakernel_likely",
    ]
    assert split_schedule.buckets[1].max_vmem_tokens == definite_merge.max_vmem_tokens

    ctx_low = dataclasses.replace(
        base_entry,
        active_tokens=16,
        effective_context=4096,
        recommended_scope="whole_step",
        verdict="megakernel_likely",
    )
    ctx_high = dataclasses.replace(
        base_entry,
        active_tokens=16,
        effective_context=32768,
        recommended_scope="per_op",
        verdict="separate_likely",
    )
    ctx_schedule = advisor.build_bucket_schedule(
        dataclasses.replace(base_sweep, regimes=(ctx_low, ctx_high))
    )
    assert [bucket.effective_context for bucket in ctx_schedule.buckets] == [
        4096,
        32768,
    ]

    qwen_schedule = advisor.build_bucket_schedule(base_sweep)
    qwen_decode_buckets = [
        bucket
        for bucket in qwen_schedule.buckets
        if bucket.phase == "decode" and bucket.layout == base_sweep.parallelism.label()
    ]
    assert len(qwen_decode_buckets) >= MIN_SWEEP_REGIMES
    ready_bucket = next(
        bucket
        for bucket in qwen_decode_buckets
        if bucket.core_support == "within_core_limits"
    )
    blocked_bucket = next(
        bucket
        for bucket in qwen_decode_buckets
        if bucket.core_support == "outside_core_limits"
    )
    assert ready_bucket.recommended_scope == "whole_step"
    assert ready_bucket.core_ready_scope == "whole_step"
    assert ready_bucket.core_blockers == ()
    assert blocked_bucket.recommended_scope == "whole_step"
    assert blocked_bucket.core_ready_scope == "per_op"
    assert "row_limit_not_stated" in blocked_bucket.core_blockers


def test_cli_html_and_bucket_schedule_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CLI `--html`, `--bucket-schedule`, valid combos, and collision errors."""
    sweep_html = tmp_path / "sweep.html"
    sweep_exit_code = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--regime-sweep",
            "--html",
            str(sweep_html),
        ]
    )
    assert sweep_exit_code == 0
    sweep_stdout = capsys.readouterr().out
    assert "=== Megakernel Advisor Regime Report: qwen3.5-397b" in sweep_stdout
    sweep_html_text = sweep_html.read_text(encoding="utf-8")
    assert "<svg " in sweep_html_text
    assert "<script" not in sweep_html_text.lower()

    single_html = tmp_path / "single.html"
    single_exit_code = advisor.main(
        [
            "estimate",
            "--model",
            "qwen3.5-397b",
            "--phase",
            "decode",
            "--batch",
            "16",
            "--tp",
            "8",
            "--ep",
            "8",
            "--html",
            str(single_html),
        ]
    )
    assert single_exit_code == 0
    single_stdout = capsys.readouterr().out
    assert "Model: qwen3.5-397b" in single_stdout
    single_html_text = single_html.read_text(encoding="utf-8")
    assert "<svg " in single_html_text
    assert "<script" not in single_html_text.lower()

    schedule_path = tmp_path / "bucket_schedule.json"
    schedule_file_exit_code = advisor.main(
        ["--model", "qwen3.5-397b", "--bucket-schedule", str(schedule_path)]
    )
    assert schedule_file_exit_code == 0
    assert capsys.readouterr().out == ""
    schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    assert schedule_payload["schema"] == advisor.BUCKET_SCHEDULE_SCHEMA

    combo_schedule_path = tmp_path / "combo_schedule.json"
    combo_exit_code = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--regime-sweep",
            "--bucket-schedule",
            str(combo_schedule_path),
        ]
    )
    assert combo_exit_code == 0
    assert (
        "=== Megakernel Advisor Regime Report: qwen3.5-397b" in capsys.readouterr().out
    )
    assert combo_schedule_path.is_file()

    schedule_stdout_exit_code = advisor.main(
        ["--model", "qwen3.5-397b", "--bucket-schedule", "-"]
    )
    assert schedule_stdout_exit_code == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["consumer_status"] == advisor.BUCKET_CONSUMER_STATUS

    invalid_html = tmp_path / "invalid_tp.html"
    invalid_exit_code = advisor.main(
        [
            "--model",
            "qwen3.5-397b",
            "--tp",
            str(SOFT_INVALID_TP_DEGREE),
            "--ep",
            "1",
            "--html",
            str(invalid_html),
        ]
    )
    assert invalid_exit_code == advisor.constants.INVALID_CONFIG_EXIT_CODE
    capsys.readouterr()
    assert "INVALID CONFIG" in invalid_html.read_text(encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    out_html = tmp_path / "blocked.html"
    out_sched = tmp_path / "blocked.json"
    for colliding_argv in (
        ["--model", "qwen3.5-397b", "--bucket-schedule", "-", "--json", "-"],
        ["--model", "qwen3.5-397b", "--bucket-schedule", "-", "--regime-sweep"],
        ["--model", "qwen3.5-397b", "--search-layer-groups", "--html", str(out_html)],
        [
            "--model",
            "qwen3.5-397b",
            "--search-layer-groups",
            "--bucket-schedule",
            str(out_sched),
        ],
        ["--model", "qwen3.5-397b", "--search-layer-groups", "--regime-sweep"],
        ["--list-models", "--html", str(out_html)],
        ["--list-models", "--bucket-schedule", str(out_sched)],
        ["--model", "qwen3.5-397b", "--html", "-"],
    ):
        with pytest.raises(errors.MegakernelConfigError):
            advisor.main(colliding_argv)
    assert not out_html.exists()
    assert not out_sched.exists()
    assert not (tmp_path / "-").exists()


@pytest.mark.parametrize(
    "conflicting_flags",
    [
        ["--search-layer-groups"],
        ["--explain"],
        ["--batch", "64"],
        ["--phase", "prefill"],
        ["--layer-groups", "4,4"],
        ["--moe-mode", "grouped_on_chip"],
        ["--kv-split", "2"],
        [
            "--batch",
            "64",
            "--phase",
            "prefill",
            "--layer-groups",
            "4,4",
            "--explain",
        ],
    ],
)
def test_bucket_schedule_rejects_conflicting_single_run_flags(
    tmp_path: Path, conflicting_flags: list[str]
) -> None:
    """Verify `--bucket-schedule` fails closed when paired with single-run flags."""
    out_sched = tmp_path / "should_not_exist.json"
    with pytest.raises(errors.MegakernelConfigError):
        advisor.main(
            [
                "--model",
                "qwen3.5-397b",
                "--bucket-schedule",
                str(out_sched),
                *conflicting_flags,
            ]
        )
    assert not out_sched.exists()


def test_unequal_chip_layout_comparison_formatting_invariants() -> None:
    """Verify unequal-chip non-comparability and axis formatting across outputs."""
    unequal_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            context=4096,
            tp=16,
            ep=16,
            objective="both",
        ),
        "tpu7x",
    )
    assert "layout_not_comparable" in unequal_report.flags
    single_html = advisor.render_advisor_html(unequal_report)
    assert unequal_report.rationale in single_html
    assert "layout_not_comparable" in single_html

    unequal_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(objective="both"),
            tp=16,
            ep=16,
        ),
    )
    sweep_html = advisor.render_advisor_html(unequal_sweep)
    assert any(
        "layout_not_comparable" in entry.flags and entry.rationale in sweep_html
        for entry in unequal_sweep.regimes
    )
    assert "layout_not_comparable" in sweep_html
    schedule = advisor.build_bucket_schedule(unequal_sweep)
    assert UNIT_AXIS_RE.search(schedule.default_layout) is None
    for bucket in schedule.buckets:
        assert UNIT_AXIS_RE.search(bucket.layout) is None


def test_calibrated_e2e_gain_basis_and_spelled_out_model_facts() -> None:
    """Verify single/sweep calibrated E2E basis and spelled-out model facts."""
    trace_payload = {
        "schema": "advisor.measurements.v1",
        "source": {"kind": "xplane", "file": "calibrated_e2e.xplane.pb"},
        "launch_gap_us": {"p50": 1.2, "p90": 1.5, "n": 40},
        "step_time_us": {"p50": TRACE_STEP_TIME_US, "n": 40},
        "step_coverage": {"fused_eligible_fraction": 0.9, "basis": "trace"},
    }
    single_report = advisor.advise(
        "kimi-k3",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=32, ep=32),
        "tpu7x",
        measurements=trace_payload,
    )
    assert single_report.gain_basis == "calibrated_whole_server_e2e"
    single_html = advisor.render_advisor_html(single_report)
    assert "Basis: calibrated whole-server E2E" in single_html
    assert "Basis: <code>calibrated_whole_server_e2e" not in single_html

    sweep = advisor.advise_regimes("kimi-k3", "tpu7x", measurements=trace_payload)
    assert any(
        entry.gain_basis == "calibrated_whole_server_e2e" for entry in sweep.regimes
    )
    html_out = advisor.render_advisor_html(sweep)
    assert "basis: calibrated whole-server E2E" in html_out
    assert "attn_heads=" in html_out
    assert "kv_heads=" in html_out
    assert "head_dim=" in html_out
    assert "experts=" in html_out
    assert "hidden=" in html_out
    assert sweep.model_profile is not None
    assert f"state={sweep.model_profile.state_dtype}" in html_out
    for abbreviation in ("NH=", "NKV=", "HD=", " E="):
        assert abbreviation not in html_out

    dense_profile = dataclasses.replace(
        advisor.resolve_model_profile("qwen3.5-397b"),
        num_experts=0,
        top_k=0,
        moe_intermediate_size=0,
        num_moe_layers=0,
    )
    dense_sweep = dataclasses.replace(
        sweep,
        model_profile=dense_profile,
        model_sources={**sweep.model_sources, "dtypes": "dense bf16/fp8"},
    )
    dense_html = advisor.render_advisor_html(dense_sweep)
    assert "Dense MLP" in dense_html
    assert "expert=" not in dense_html


def test_decision_row_surfaces_physics_winner_and_core_readiness() -> None:
    """Verify KPIs, Decision Table, and unit labels separate physics from core."""
    qwen_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    qwen_small = next(
        entry for entry in qwen_sweep.regimes if entry.label.startswith("Decode Small")
    )
    qwen_high = next(
        entry for entry in qwen_sweep.regimes if entry.label.startswith("Decode High")
    )
    assert qwen_small.recommended_scope == "whole_step"
    assert qwen_small.core_support == "within_core_limits"
    assert qwen_small.core_ready_scope == "whole_step"
    assert qwen_small.core_blockers == ()
    assert qwen_high.recommended_scope == "whole_step"
    assert qwen_high.core_support == "outside_core_limits"
    assert qwen_high.core_ready_scope == "per_op"
    assert "row_limit_not_stated" in qwen_high.core_blockers

    qwen_html = advisor.render_advisor_html(qwen_sweep)
    qwen_body = re.sub(
        r"<style\b[^>]*>.*?</style>", "", qwen_html, flags=re.DOTALL | re.IGNORECASE
    )
    assert "badge-whole_step" in qwen_body
    assert "WHOLE_STEP" in qwen_body
    assert "WHOLE-STEP MEGAKERNEL" in qwen_body
    assert "badge-within_core_limits" in qwen_body
    assert "within_core_limits" in qwen_body
    assert "badge-outside_core_limits" in qwen_body
    assert "outside_core_limits" in qwen_body
    assert "row_limit_not_stated" in qwen_body
    assert "fused core-ready" in qwen_body
    assert "baseline per_op" in qwen_body
    assert "mxu" not in qwen_body.lower()

    kimi_sweep = advisor.advise_regimes("kimi-k3", "tpu7x")
    decode_small = next(
        entry for entry in kimi_sweep.regimes if entry.label.startswith("Decode Small")
    )
    assert decode_small.recommended_scope == "whole_step"
    assert decode_small.fastest_scope == "whole_step"
    assert decode_small.core_support == "outside_core_limits"
    assert decode_small.core_ready_scope == "per_op"
    assert "kv_or_state_dtype_not_in_core" in decode_small.core_blockers
    kimi_html = advisor.render_advisor_html(kimi_sweep)
    kimi_body = re.sub(
        r"<style\b[^>]*>.*?</style>", "", kimi_html, flags=re.DOTALL | re.IGNORECASE
    )
    assert '<td class="tabular text-gain">+' in kimi_body
    assert "WHOLE_STEP" in kimi_body
    assert "outside_core_limits" in kimi_body
    assert "core_ready_scope=<code>per_op</code>" in kimi_body
    assert "kv_or_state_dtype_not_in_core" in kimi_body
    assert "mxu" not in kimi_body.lower()


def test_single_report_core_readiness_and_matmul_unit_labels() -> None:
    """Verify single-report KPIs show core readiness and map compute units."""
    qwen_single = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=8, ep=8, dcp=4),
        "tpu7x",
    )
    qwen_single_html = advisor.render_advisor_html(qwen_single)
    assert "WHOLE_STEP" in qwen_single_html
    assert "within_core_limits" in qwen_single_html

    kimi_single = advisor.advise(
        "kimi-k3",
        advisor.WorkloadSpec(phase="decode", batch=4, tp=32, ep=32, dcp=32),
        "tpu7x",
    )
    kimi_single_html = advisor.render_advisor_html(kimi_single)
    assert "WHOLE_STEP" in kimi_single_html
    assert "outside_core_limits" in kimi_single_html
    assert "core_ready_scope=<code>per_op</code>" in kimi_single_html
    assert "kv_or_state_dtype_not_in_core" in kimi_single_html

    prefill_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="prefill", isl=4096, tp=1, pp=8),
        "tpu7x",
    )
    prefill_html = advisor.render_advisor_html(prefill_report)
    assert "<code>matmul</code>" in prefill_html
    assert "mxu" not in prefill_html.lower()


def test_html_hbm_overflow_oom_and_invalid_config_badges() -> None:
    """Verify pool-cap overflow, physical OOM, and invalid config verdict badges."""
    report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            tp=8,
            ep=8,
            hbm_utilization_cap=POOL_CAP_FRACTION,
        ),
        "tpu7x",
    )
    assert report.hbm_pool_overflow is True
    single_html = advisor.render_advisor_html(report)
    assert "exceeds 45% pool cap" in single_html
    assert "exceeds 92% pool cap" not in single_html

    sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(hbm_utilization_cap=POOL_CAP_FRACTION)
        ),
    )
    sweep_html = advisor.render_advisor_html(sweep)
    assert "exceeds 45% pool cap" in sweep_html

    oom_report = advisor.advise(
        "kimi-k3",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    assert oom_report.hbm_fits is False
    oom_html = advisor.render_advisor_html(oom_report)
    assert "PHYSICAL OOM" in oom_html

    invalid_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=SOFT_INVALID_TP_DEGREE, ep=1),
        "tpu7x",
    )
    invalid_single_html = advisor.render_advisor_html(invalid_report)
    assert "verdict-invalid" in invalid_single_html
    assert "INVALID CONFIG" in invalid_single_html
    assert "SEPARATE OPS" not in invalid_single_html

    invalid_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=SOFT_INVALID_TP_DEGREE, ep=1),
    )
    invalid_sweep_html = advisor.render_advisor_html(invalid_sweep)
    invalid_sweep_body = re.sub(
        r"<style\b[^>]*>.*?</style>",
        "",
        invalid_sweep_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert "INVALID CONFIG" in invalid_sweep_body
    assert "verdict-separate" not in invalid_sweep_body


@pytest.mark.parametrize(
    "missing_key",
    ["launch_gap_us", "collective_latency_us", "step_coverage"],
)
def test_render_html_raises_when_required_prior_metrics_missing(
    missing_key: str,
) -> None:
    """Verify facts ledger fails closed when any required prior metric is absent."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    stripped_metrics = {
        key: metric_value
        for key, metric_value in sweep.regimes[0].metrics.items()
        if key != missing_key
    }
    broken_entry = dataclasses.replace(sweep.regimes[0], metrics=stripped_metrics)
    broken_sweep = dataclasses.replace(
        sweep, regimes=(broken_entry, *sweep.regimes[1:])
    )
    with pytest.raises(KeyError):
        advisor.render_advisor_html(broken_sweep)


MIN_SWEEP_SVGS = 3
MIN_SWEEP_REGIMES = 2


def _parse_strict_xml_elements(
    svg_markup: str,
) -> list[tuple[str, dict[str, str], str]]:
    """Parse one `<svg>` document via strict Expat XML into element tuples."""
    elements: list[tuple[str, dict[str, str], str]] = []
    stack: list[tuple[str, dict[str, str], list[str]]] = []

    def _on_start(name: str, attrs: dict[str, str]) -> None:
        """Push an opening XML element onto the parser stack."""
        stack.append((name.rsplit(":", 1)[-1], dict(attrs), []))

    def _on_data(chunk: str) -> None:
        """Accumulate character data for the current XML element."""
        if stack:
            stack[-1][2].append(chunk)

    def _on_end(_name: str) -> None:
        """Pop and record a closed XML element."""
        tag, attrs, chunks = stack.pop()
        elements.append((tag, attrs, "".join(chunks)))

    parser = expat.ParserCreate()
    parser.StartElementHandler = _on_start
    parser.CharacterDataHandler = _on_data
    parser.EndElementHandler = _on_end
    is_final = True
    parser.Parse(svg_markup, is_final)
    return elements


@dataclasses.dataclass
class _TableParityState:
    """Mutable column-count tracking state for one parsed HTML table."""

    in_thead: bool = False
    in_tbody: bool = False
    thead_cols: int = 0
    tbody_rows: list[int] = dataclasses.field(default_factory=list)
    cur_row: int | None = None


class _HtmlTableParityParser(HTMLParser):
    """Parse HTML tables and verify every tbody row matches thead column count."""

    def __init__(self) -> None:
        """Initialize parser state for nested HTML table column parity checks."""
        super().__init__(convert_charrefs=True)
        self.tables: list[tuple[int, list[int]]] = []
        self._stack: list[_TableParityState] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track nested tables, thead/tbody sections, and cell colspans."""
        attr_map = {k: (v or "") for k, v in attrs}
        if tag == "table":
            self._stack.append(_TableParityState())
            return
        if not self._stack:
            return
        top = self._stack[-1]
        if tag == "thead":
            top.in_thead = True
        elif tag == "tbody":
            top.in_tbody = True
        elif tag == "tr" and top.in_tbody:
            top.cur_row = 0
        elif tag == "th" and top.in_thead:
            top.thead_cols += int(attr_map.get("colspan") or "1")
        elif tag in ("td", "th") and top.cur_row is not None:
            top.cur_row += int(attr_map.get("colspan") or "1")

    def handle_endtag(self, tag: str) -> None:
        """Finalize rows, sections, and completed tables."""
        if not self._stack:
            return
        top = self._stack[-1]
        if tag == "thead":
            top.in_thead = False
        elif tag == "tbody":
            top.in_tbody = False
        elif tag == "tr" and top.cur_row is not None:
            top.tbody_rows.append(top.cur_row)
            top.cur_row = None
        elif tag == "table":
            done = self._stack.pop()
            self.tables.append((done.thead_cols, done.tbody_rows))


def _assert_table_column_parity(html_doc: str) -> list[tuple[int, list[int]]]:
    """Assert every HTML table thead column count matches each tbody row."""
    parser = _HtmlTableParityParser()
    parser.feed(html_doc)
    parser.close()
    assert parser.tables
    for thead_cols, tbody_rows in parser.tables:
        if thead_cols > 0:
            for row_cols in tbody_rows:
                assert row_cols == thead_cols, (
                    f"Table column mismatch: thead={thead_cols} vs tbody row={row_cols}"
                )
    return parser.tables


def _assert_no_same_y_text_overlap(svg_markup: str) -> None:
    """Assert same-y SVG text labels do not horizontally overlap."""
    by_y: dict[float, list[tuple[float, float, str]]] = {}
    for tag, attrs, raw_text in _parse_strict_xml_elements(svg_markup):
        if tag != "text":
            continue
        label = raw_text.strip()
        if not label:
            continue
        x_val = float(attrs.get("x", "0"))
        y_val = round(float(attrs.get("y", "0")), 1)
        font_px = float(attrs.get("font-size", "13"))
        width_px = advisor_report_mod.svg_text_width(label, font_px)
        anchor = attrs.get("text-anchor", "start")
        if anchor == "middle":
            x_left, x_right = x_val - width_px / 2.0, x_val + width_px / 2.0
        elif anchor == "end":
            x_left, x_right = x_val - width_px, x_val
        else:
            x_left, x_right = x_val, x_val + width_px
        by_y.setdefault(y_val, []).append((x_left, x_right, label))
    for y_val, boxes in by_y.items():
        boxes.sort(key=lambda item: item[0])
        for prev_box, cur_box in itertools.pairwise(boxes):
            assert prev_box[1] <= cur_box[0] + 1.0, (
                f"Overlapping SVG text at y={y_val}: {prev_box} vs {cur_box}"
            )


def test_html_strict_xml_svgs_and_table_column_parity() -> None:
    """Verify strict XML SVGs, zero JS, and thead/tbody parity across presets."""
    expected_decision_cols = advisor.constants.DECISION_TABLE_COLS
    for model_slug in sorted(advisor.MODEL_PRESETS):
        for chip_name in ("tpu7x", "v6e"):
            sweep = advisor.advise_regimes(model_slug, chip_name)
            sweep_html = advisor.render_advisor_html(sweep)
            assert "<script" not in sweep_html.lower()
            svgs = SVG_BLOCK_RE.findall(sweep_html)
            assert len(svgs) >= MIN_SWEEP_SVGS
            for svg_markup in svgs:
                _parse_strict_xml_elements(svg_markup)
            tables = _assert_table_column_parity(sweep_html)
            assert any(cols == expected_decision_cols for cols, _ in tables)

            profile = advisor.resolve_model_profile(model_slug)
            single_report = advisor.advise(
                profile,
                advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
                chip_name,
            )
            single_html = advisor.render_advisor_html(single_report)
            assert "<script" not in single_html.lower()
            single_svgs = SVG_BLOCK_RE.findall(single_html)
            assert len(single_svgs) >= 1
            for svg_markup in single_svgs:
                _parse_strict_xml_elements(svg_markup)
            _assert_table_column_parity(single_html)


def test_facts_card_aggregates_measured_across_all_regimes() -> None:
    """Verify top facts card shows CALIBRATED TRACE for non-first regimes."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    assert len(sweep.regimes) >= MIN_SWEEP_REGIMES
    source_tag = "non_first_row_trace.xplane.pb"
    second_metrics = dict(sweep.regimes[1].metrics)
    second_metrics["collective_call_us_xla"] = {
        "value": 12.5,
        "unit": "us",
        "tag": "MEASURED",
        "band": [12.5, 12.5],
        "source": source_tag,
    }
    patched_second = dataclasses.replace(sweep.regimes[1], metrics=second_metrics)
    patched_sweep = dataclasses.replace(
        sweep,
        regimes=(sweep.regimes[0], patched_second, *sweep.regimes[2:]),
    )
    html_doc = advisor.render_advisor_html(patched_sweep)
    assert "CALIBRATED TRACE" in html_doc
    assert source_tag in html_doc
    assert "UNCALIBRATED PRIORS" not in html_doc


def test_report_fails_closed_on_missing_or_invalid_metrics() -> None:
    """Verify missing or non-numeric report metrics raise KeyError."""
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    required_keys = (
        "compute_bound_at",
        "throughput_tok_s_per_chip",
        "fusion_gain_max",
        "compiler_proof_gain_us",
        "dispatch_bubble_gain_us",
        "compiler_proof_gain_pct",
        "break_even_host_head_delta_us",
        "launch_gap_us",
        "collective_latency_us",
        "step_coverage",
    )
    for key in required_keys:
        missing_metrics = {
            metric_key: metric_val
            for metric_key, metric_val in sweep.regimes[0].metrics.items()
            if metric_key != key
        }
        missing_sweep = dataclasses.replace(
            sweep,
            regimes=(
                dataclasses.replace(sweep.regimes[0], metrics=missing_metrics),
                *sweep.regimes[1:],
            ),
        )
        with pytest.raises(KeyError):
            advisor.render_advisor_html(missing_sweep)

        for bad_val in ("not_a_number", None):
            bad_metrics = dict(sweep.regimes[0].metrics)
            corrupted_entry = typing.cast(
                "advisor_specs.ProvenanceMetric",
                {**bad_metrics[key], "value": bad_val},
            )
            bad_metrics[key] = corrupted_entry
            bad_sweep = dataclasses.replace(
                sweep,
                regimes=(
                    dataclasses.replace(sweep.regimes[0], metrics=bad_metrics),
                    *sweep.regimes[1:],
                ),
            )
            with pytest.raises(KeyError):
                advisor.render_advisor_html(bad_sweep)


def test_vmem_chart_conservation_pcp_invariance_and_label_layout() -> None:
    """Verify Chart 1 VMEM fit, decode PCP invariance, and non-overlapping text."""
    for model_slug in sorted(advisor.MODEL_PRESETS):
        for headroom in (False, True):
            for row_cap in (0, CUSTOM_ROW_CAP):
                sweep = advisor.advise_regimes(
                    model_slug,
                    "tpu7x",
                    options=advisor.RegimeSweepOptions(
                        base_workload=advisor.WorkloadSpec(
                            vmem_fit_headroom=headroom,
                            max_rows_per_call=row_cap,
                        )
                    ),
                )
                svg_markup = SVG_BLOCK_RE.findall(advisor.render_advisor_html(sweep))[0]
                elements = _parse_strict_xml_elements(svg_markup)
                _assert_no_same_y_text_overlap(svg_markup)
                for tag, _attrs, raw_text in elements:
                    if tag != "text":
                        continue
                    match = BAR_LABEL_RE.match(raw_text.strip())
                    if match and int(match.group(2)) == 1:
                        total_mib = float(match.group(1))
                        assert (
                            total_mib <= sweep.vmem_budget_mib + 0.1 + FLOAT_TOLERANCE
                        )

    # Decode PCP invariance under effective_pcp == 1: pcp=2 matches pcp=1.
    sweep_pcp1 = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=8, ep=8, pcp=1),
    )
    sweep_pcp2 = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(tp=8, ep=8, pcp=2),
    )
    svg_pcp1 = SVG_BLOCK_RE.findall(advisor.render_advisor_html(sweep_pcp1))[0]
    svg_pcp2 = SVG_BLOCK_RE.findall(advisor.render_advisor_html(sweep_pcp2))[0]
    assert svg_pcp1 == svg_pcp2

    # Prefill-only custom budget persistence & empty-regime headroom row-cap accuracy.
    custom_budget_mib = 30.0
    base_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(vmem_budget_mib=custom_budget_mib)
        ),
    )
    prefill_only = dataclasses.replace(
        base_sweep,
        regimes=tuple(
            entry for entry in base_sweep.regimes if entry.phase == "prefill"
        ),
    )
    assert prefill_only.regimes
    prefill_svg = SVG_BLOCK_RE.findall(advisor.render_advisor_html(prefill_only))[0]
    full_svg = SVG_BLOCK_RE.findall(advisor.render_advisor_html(base_sweep))[0]
    assert prefill_svg == full_svg

    headroom_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(vmem_fit_headroom=True)
        ),
    )
    empty_headroom = dataclasses.replace(headroom_sweep, regimes=())
    empty_svg = SVG_BLOCK_RE.findall(advisor.render_advisor_html(empty_headroom))[0]
    assert "row cap <" not in empty_svg

    # Tiny VMEM budget (max_vmem_tokens == 0) prints "VMEM OOM", never "4 passes".
    oom_sweep = advisor.advise_regimes(
        "qwen3.5-397b",
        "tpu7x",
        options=advisor.RegimeSweepOptions(
            base_workload=advisor.WorkloadSpec(vmem_budget_mib=2.0)
        ),
    )
    assert oom_sweep.max_vmem_tokens == 0
    oom_svg = SVG_BLOCK_RE.findall(advisor.render_advisor_html(oom_sweep))[0]
    assert "VMEM OOM" in oom_svg
    assert "(4 passes" not in oom_svg


def test_band_charts_geometry_gridlines_and_sign_formatting() -> None:
    """Verify Chart 3/4 whiskers, narrow-band gridlines, and sign formatting."""
    for model_slug in sorted(advisor.MODEL_PRESETS):
        sweep = advisor.advise_regimes(model_slug, "tpu7x")
        chart3_svg = advisor_report_mod.render_step_band_svg(sweep)
        _parse_strict_xml_elements(chart3_svg)
        assert "+-" not in chart3_svg

        profile = advisor.resolve_model_profile(model_slug)
        single = advisor.advise(
            profile,
            advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
            "tpu7x",
        )
        single_svgs = SVG_BLOCK_RE.findall(advisor.render_advisor_html(single))
        assert single_svgs
        chart4_svg = single_svgs[0]
        _parse_strict_xml_elements(chart4_svg)
        assert "+-" not in chart4_svg
        for match in WHISKER_RE.finditer(chart4_svg):
            attrs = dict(ATTR_RE.findall(match.group(1)))
            assert float(attrs["x1"]) <= float(attrs["x2"])

    # Narrow band [51_000, 86_000] (no 1-2-5 decade ticks) emits >= 1 gridline
    # and formats negative per_op delta without '+-'.
    base_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    base_entry = base_sweep.regimes[0]
    narrow_scopes = tuple(
        {
            **scope_dict,
            "step_bound_us": 55_000.0
            if scope_dict.get("scope") == "per_op"
            else 60_000.0,
            "step_upper_us": 72_000.0,
            "step_floor_strict_us": 51_000.0,
            "step_achievable_us": 65_000.0,
        }
        for scope_dict in base_entry.scopes
    )
    narrow_entry = dataclasses.replace(
        base_entry,
        recommended_scope="whole_step",
        step_bound_us=60_000.0,
        step_upper_us=72_000.0,
        step_floor_strict_us=51_000.0,
        step_achievable_us=65_000.0,
        scopes=narrow_scopes,
    )
    narrow_sweep = dataclasses.replace(base_sweep, regimes=(narrow_entry,))
    narrow_svg = advisor_report_mod.render_step_band_svg(narrow_sweep)
    elements = _parse_strict_xml_elements(narrow_svg)
    assert "+-" not in narrow_svg
    grid_lines = [
        attrs
        for tag, attrs, _text in elements
        if tag == "line" and attrs.get("stroke-dasharray") == "3,3"
    ]
    assert len(grid_lines) >= 1


def test_bucket_schedule_context_partitioning_and_exclusion() -> None:
    """Verify bucket schedule excludes OOM 4-tuples and keeps context partitions."""
    for model_slug in sorted(advisor.MODEL_PRESETS):
        for chip_name in ("tpu7x", "v6e"):
            sweep = advisor.advise_regimes(model_slug, chip_name)
            schedule = advisor.build_bucket_schedule(sweep)
            oom_keys = {
                (e.phase, e.layout, e.active_tokens, e.effective_context)
                for e in sweep.regimes
                if not e.hbm_fits or e.verdict.startswith("invalid_config(")
            }
            for bucket in schedule.buckets:
                assert not bucket.verdict.startswith("invalid_config(")
                for tok in (bucket.min_active_tokens, bucket.max_active_tokens):
                    assert (
                        bucket.phase,
                        bucket.layout,
                        tok,
                        bucket.effective_context,
                    ) not in oom_keys
            encoded = json.loads(json.dumps(schedule.to_dict()))
            assert encoded == schedule.to_dict()


def test_report_cli_rejects_empty_dir_and_bad_flag_combos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify `--html` and `--bucket-schedule` reject bad paths and conflicts."""
    cwd_dir = tmp_path / "cwd_report_clean"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)
    out_sched = tmp_path / "sched.json"
    out_html = tmp_path / "report.html"
    bad_argvs: tuple[list[str], ...] = (
        ["--model", "qwen3.5-397b", "--html", ""],
        ["--model", "qwen3.5-397b", "--html", "   "],
        ["--model", "qwen3.5-397b", "--html", str(cwd_dir)],
        ["--model", "qwen3.5-397b", "--html", "-"],
        ["--model", "qwen3.5-397b", "--html", " - "],
        ["--model", "qwen3.5-397b", "--bucket-schedule", ""],
        ["--model", "qwen3.5-397b", "--bucket-schedule", "   "],
        ["--model", "qwen3.5-397b", "--bucket-schedule", str(cwd_dir)],
        [
            "--model",
            "qwen3.5-397b",
            "--bucket-schedule",
            str(out_sched),
            "--unresolved",
        ],
        [
            "--model",
            "qwen3.5-397b",
            "--tp",
            "8",
            "--ep",
            "8",
            "--search-layer-groups",
            "--html",
            str(out_html),
        ],
    )
    for argv in bad_argvs:
        with pytest.raises(errors.MegakernelConfigError):
            advisor.main(argv)
    assert list(cwd_dir.iterdir()) == []
