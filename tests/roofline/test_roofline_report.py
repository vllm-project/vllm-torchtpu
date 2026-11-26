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
"""CPU unit tests for Layer-1 per-concurrency offline roofline HTML reports."""

from __future__ import annotations

import copy
import html as html_lib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.roofline.conftest import LOCKED_FIXTURE_FILES, load_locked_bundle
from vllm_torchtpu.roofline import (
    CHIPS,
    SLO_VERDICT_RULED_OUT,
    SolveOptions,
    TrafficScenario,
    build_solve_payload,
    cli,
    render_html,
    solve_serving_plan,
    write_html,
)
from vllm_torchtpu.roofline import report as report_module
from vllm_torchtpu.roofline.constants import (
    FRONTIER_2D_TIE_TOLERANCE,
    MXFP4_BYTES,
    UNKNOWN_THINK_TIME_WARNING,
)

if TYPE_CHECKING:
    from vllm_torchtpu.roofline import SolvePayload


@pytest.fixture(scope="module")
def sample_payload() -> SolvePayload:
    """Return a valid physics solve payload derived from a locked basis fixture."""
    bundle = load_locked_bundle("qwen3_5_397b_fp8.json")
    chip = CHIPS["tpu7x"]
    scenario = TrafficScenario(
        name="chat-8k1k",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=(1, 2),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="auto", max_chips=4),
    )
    return build_solve_payload(
        plan,
        chip=chip,
        dtypes=bundle.dtypes,
        model=bundle.identity.hf_repo,
        top_k=3,
    )


def test_report_html_is_deterministic(sample_payload: SolvePayload) -> None:
    """render_html produces byte-identical output across consecutive calls."""
    first_html = render_html(sample_payload)
    second_html = render_html(sample_payload)
    min_html_len = 500
    assert first_html == second_html
    assert len(first_html) > min_html_len


def test_report_has_no_external_fetch(sample_payload: SolvePayload) -> None:
    """HTML report is fully offline: zero external stylesheets, scripts, or fetches."""
    html = render_html(sample_payload)
    assert "<link" not in html
    assert not re.search(r"<script[^>]+src=", html, re.IGNORECASE)
    assert not re.search(r"""(?:src|href)\s*=\s*["']https?://""", html, re.IGNORECASE)
    assert not re.search(r"""url\(\s*["']?https?://""", html, re.IGNORECASE)
    assert "@import" not in html
    found_urls = re.findall(r"http://[a-zA-Z0-9./_-]+", html)
    assert set(found_urls).issubset({"http://www.w3.org/2000/svg"})


def test_report_escapes_untrusted_strings(sample_payload: SolvePayload) -> None:
    """Model names and warning text with HTML or script tags are properly escaped."""
    payload: dict[str, object] = dict(sample_payload)
    payload["model"] = '<script>alert("xss")</script>'
    payload["warnings"] = ["<b>Unsafe warning</b>", "Normal warning"]
    html = render_html(payload)
    assert '<script>alert("xss")</script>' not in html
    assert (
        "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html
        or "&lt;script&gt;" in html
    )
    assert "<b>Unsafe warning</b>" not in html
    assert "&lt;b&gt;Unsafe warning&lt;/b&gt;" in html
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    json_blob = match.group(1)
    assert "</" not in json_blob or r"<\/" in json_blob


def test_report_sections_present_pure_l1_and_zero_launch_boxes(
    sample_payload: SolvePayload,
) -> None:
    """Required pure-L1 sections, controls, and pool tables exist with zero launch."""
    html = render_html(sample_payload)
    assert 'id="physical-envelope"' in html
    assert 'id="search-funnel"' in html
    assert 'id="concurrency-controls"' in html
    assert 'id="frontier-chart"' in html
    assert 'id="candidate-explorer"' in html
    assert 'id="limits-chart"' in html
    assert 'id="memory-chart"' in html
    assert 'id="pool-tables"' in html
    assert 'id="roofline-data"' in html
    assert 'data-plot-btn="est"' in html
    assert 'data-plot-btn="bound"' in html
    assert 'id="explorer-sort"' in html
    assert 'id="explorer-filter"' in html
    assert "<svg" in html
    assert "shortlist" not in html.lower()
    assert "Est. 2D Frontier" not in html
    assert "copyLaunch" not in html
    assert "vllm serve" not in html
    assert 'details class="launch"' not in html
    assert "state=float32" in html
    assert "All Legal Layouts" not in html
    assert "__AXIS" not in html
    assert "__VISIBLE_ROWS__" not in html
    assert "var NICE_STEPS = [" in html
    assert 'id="explorer-row-controls"' in html
    assert 'id="memory-legend"' in html
    assert html.count("Free Headroom") == 1
    for term in (
        "Binding:",
        "Tput Ceiling [bound]:",
        "KV Cap / Prefix Hit Rate:",
        "Prefix-Hit Regime [est]:",
    ):
        assert f"<strong>{term}</strong>" in html


def _embedded_report_payload(html: str) -> dict[str, object]:
    """Parse the JSON payload embedded in a rendered report."""
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    return json.loads(match.group(1).replace(r"<\/", "</"))


def _assert_plotted_points_inside_frame(chart: str) -> None:
    """Assert every plotted layout circle lies inside the chart's axis frame."""
    frame = re.search(
        r'<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)" fill="#f8fafc"', chart
    )
    assert frame is not None
    left, top, width, height = (float(val) for val in frame.groups())
    circles = re.findall(
        r'<circle cx="([0-9.]+)" cy="([0-9.]+)" r="[0-9.]+" fill="#', chart
    )
    assert circles
    for center_x, center_y in circles:
        assert left <= float(center_x) <= left + width
        assert top <= float(center_y) <= top + height


def _assert_axis_ticks_distinct_and_evenly_spaced(chart: str) -> None:
    """Assert each axis prints 5 different tick labels one constant step apart."""
    for anchor in ("middle", "end"):
        labels = re.findall(
            rf'text-anchor="{anchor}" font-size="11.5" fill="#64748b">([^<]*)</text>',
            chart,
        )
        assert len(labels) == len(set(labels)) == 5
        values = [float(label.replace(",", "")) for label in labels]
        steps = [
            upper - lower for lower, upper in zip(values, values[1:], strict=False)
        ]
        assert min(steps) > 0.0
        assert max(steps) - min(steps) <= 1e-9 * max(steps)


def _candidates_tab_count(html: str) -> str:
    """Return the count text inside the candidates tab button of a rendered report."""
    tab_match = re.search(r'data-tab-btn="candidates">[^(<]*\(([^)]*)\)</button>', html)
    assert tab_match is not None
    return tab_match.group(1)


def _payload_with_verdicts(
    sample_payload: SolvePayload,
    verdicts: Sequence[str],
) -> dict[str, object]:
    """Return a one-concurrency payload whose layouts carry the given SLO verdicts."""
    base = copy.deepcopy(dict(sample_payload["frontier"][0]))
    points = []
    for idx, verdict in enumerate(verdicts):
        point = copy.deepcopy(base)
        point["concurrency"] = 64
        point["slo_verdict"] = verdict
        point["decode_pool"]["kv_split"] = idx + 1
        point["bound_tpot_ms"] = 0.2 + idx / 100.0
        points.append(point)
    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    payload["candidates"] = points
    payload["frontier"] = []
    payload["top_k"] = []
    payload["recommended"] = points[0]
    payload.pop("concurrency_top_k", None)
    payload.pop("bound_families_by_concurrency", None)
    return payload


def _disagg_solve_argv(cfg_path: Path) -> list[str]:
    """Return CLI arguments for a small pinned disaggregated solve."""
    return [
        "solve",
        "--config",
        str(cfg_path),
        "--isl",
        "1024",
        "--osl",
        "128",
        "--prefix-hit-rate",
        "0.0",
        "--accept-len",
        "1.0",
        "--concurrency",
        "4",
        "--mode",
        "disagg",
        "--prefill-topology",
        "2x2x1",
        "--decode-topology",
        "2x2x1",
        "--pd",
        "1:1",
    ]


def _explorer_tbody(html: str) -> str:
    """Return the static candidate explorer table body of a rendered report."""
    match = re.search(r'<tbody id="explorer-tbody">(.*?)</tbody>', html, re.DOTALL)
    assert match is not None
    return match.group(1)


def test_report_layout_counts_agree_across_funnel_card_and_explorer(
    sample_payload: SolvePayload,
) -> None:
    """Explorer tab, envelope card, and scope note state one shown/total count."""
    html = render_html(sample_payload)
    rows = _embedded_report_payload(html)["explorer"]["rows"]
    assert sample_payload["search"] is not None
    # No SLO is set, so every HBM-fit layout counts as not ruled out.
    total = int(sample_payload["search"]["hbm_fit"])
    assert len(rows) < total
    count_label = f"{len(rows):,} of {total:,}"
    assert _candidates_tab_count(html) == count_label
    assert f"{total:,} layout-concurrency points not ruled out" in html
    assert f"Plots and lists {count_label} layout-concurrency points" in html
    assert "<code>--top-k 3</code>" in html


def test_report_counts_only_layouts_not_ruled_out_when_an_slo_is_set(
    sample_payload: SolvePayload,
) -> None:
    """With an SLO, tab, card, and note count layouts not ruled out, as the funnel."""
    verdicts = ["estimate_meets"] * 3 + ["undecided"] + ["ruled_out"] * 2
    payload = _payload_with_verdicts(sample_payload, verdicts)
    payload["scenario"] = {**dict(sample_payload["scenario"]), "tpot_slo_ms": 50.0}
    payload["top_k_limit"] = 3
    assert sample_payload["search"] is not None
    search = {**dict(sample_payload["search"]), "hbm_fit": 10, "slo_met": 5}
    search.pop("not_ruled_out", None)
    payload["search"] = {**search, "not_ruled_out": 7}

    html = render_html(payload)
    assert _candidates_tab_count(html) == "4 of 7 not ruled out"
    assert "7 layout-concurrency points not ruled out across 1 concurrency;" in html
    assert (
        "Plots and lists 4 of 7 layout-concurrency points that are not ruled out "
        "(this run used <code>--top-k 3</code>, so the payload carries the first 3 "
        "layouts per concurrency by estimate order, the solver's top-3 list and "
        "recommended layout, and the Pareto-marked layouts). "
        "2 ruled-out points are listed after them. The full list is in"
    ) in html

    # Without the not-ruled-out counter the funnel's last stage is the SLO-met count.
    payload["search"] = search
    assert _candidates_tab_count(render_html(payload)) == "4 of 5 not ruled out"


def test_report_claims_all_layouts_only_with_a_matching_funnel_count(
    sample_payload: SolvePayload,
) -> None:
    """The report says all layouts are listed only when the funnel count agrees."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 6)
    assert sample_payload["search"] is not None
    payload["search"] = {
        **dict(sample_payload["search"]),
        "hbm_fit": 6,
        "not_ruled_out": 6,
    }
    html = render_html(payload)
    assert _candidates_tab_count(html) == "6"
    assert (
        "All 6 layout-concurrency points that are not ruled out are plotted and listed."
    ) in html
    assert ' style="display:none;">Show All (6)</button>' in html

    payload["search"] = None
    html = render_html(payload)
    assert _candidates_tab_count(html) == "6"
    assert "All 6" not in html
    assert (
        "Plots and lists the 6 layout-concurrency points carried in the payload "
        "that are not ruled out."
    ) in html
    assert (
        "6 layout-concurrency points carried in the payload are not ruled out, "
        "across 1 concurrency"
    ) in html


def test_explorer_cells_round_each_value_once(
    sample_payload: SolvePayload,
) -> None:
    """A cell shows the raw value rounded once, never a rounding of a rounding."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"])
    point = payload["candidates"][0]
    # Each value sits just under a tie: rounding to an extra decimal first and
    # then again would push it up (26.5495 -> 26.55 -> 26.6).
    point["bound_tpot_ms"] = 0.27449
    point["bound_ttft_ms"] = 26.5495
    point["est_interactivity_tok_s"] = 12.349
    point["est_tpot_ms"] = 11.8949
    html = render_html(payload)
    row = _embedded_report_payload(html)["explorer"]["rows"][0]
    assert (row["bt"], row["btt"], row["ei"], row["etp"]) == (0.274, 26.55, 12.3, 11.89)
    cells = [
        re.sub(r"<[^>]+>", "", cell)
        for cell in re.findall(r"<td[^>]*>(.*?)</td>", _explorer_tbody(html), re.DOTALL)
    ]
    assert (cells[4], cells[5], cells[7], cells[9]) == (
        "0.274",
        "26.55",
        "12.3",
        "11.89",
    )
    point_title = re.search(r"<title>([^<]*)</title></circle>", html)
    assert point_title is not None
    assert "TPOT bound=0.274 ms" in point_title.group(1)
    assert point_title.group(1).endswith(", 12.3 tok/s")


def test_tradeoff_legend_is_one_text_for_static_page_and_script(
    sample_payload: SolvePayload,
) -> None:
    """The static legend is the text the inline script shows for the same mode."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets", "undecided"])
    payload["reference"] = {"found": True, "point": payload["candidates"][1]}
    for tpot_slo_ms, colors in (
        (None, "No SLO set: every plotted layout passed the HBM fit"),
        (50.0, "Blue: Est. Meets · Amber: Undecided · Grey: Ruled Out"),
    ):
        payload["scenario"] = {
            **dict(sample_payload["scenario"]),
            "tpot_slo_ms": tpot_slo_ms,
        }
        html = render_html(payload)
        legend = _embedded_report_payload(html)["explorer"]["legend"]
        static = re.search(r'<div id="tradeoff-legend"[^>]*>(.*?)</div>', html)
        assert static is not None
        assert html_lib.unescape(static.group(1)) == legend["est"]
        assert legend["est"].startswith("Dashed line: uncalibrated est_* upper")
        assert legend["bound"].startswith("Dashed line: physical bound upper")
        for mode_text in legend.values():
            assert colors in mode_text
        assert "Rings: Reference / Max Est. Tput ·" in legend["est"]
        assert "Min Bound TPOT" not in legend["est"]
        assert "Max Ceiling Tput" not in legend["est"]
        assert (
            "Rings: Reference / Min Bound TPOT / Max Ceiling Tput ·" in legend["bound"]
        )
        assert "Max Est. Tput" not in legend["bound"]

    payload["reference"] = None
    legend = _embedded_report_payload(render_html(payload))["explorer"]["legend"]
    assert "Rings: Max Est. Tput ·" in legend["est"]
    assert "Rings: Min Bound TPOT / Max Ceiling Tput ·" in legend["bound"]


def test_tradeoff_callouts_match_active_plot_mode_and_view_note(
    sample_payload: SolvePayload,
) -> None:
    """Trade-off callouts stay mode-specific and view note reports overlap."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 3)
    for idx in (0, 1):
        payload["candidates"][idx]["est_interactivity_tok_s"] = 100.0
        payload["candidates"][idx]["token_throughput_per_chip"] = 5000.0
    payload["candidates"][0]["bound_tpot_ms"] = 0.25
    payload["candidates"][0]["floor_throughput_per_chip"] = 6000.0
    payload["candidates"][1]["bound_tpot_ms"] = 0.50
    payload["candidates"][1]["floor_throughput_per_chip"] = 9000.0
    payload["candidates"][2]["bound_tpot_ms"] = 0.40
    payload["candidates"][2]["floor_throughput_per_chip"] = 7000.0
    payload["candidates"][2]["est_interactivity_tok_s"] = 200.0
    payload["candidates"][2]["token_throughput_per_chip"] = 4000.0
    payload["reference"] = {"found": True, "point": payload["candidates"][2]}

    html = render_html(payload)
    tradeoff_svg = re.search(
        r'<div id="tradeoff-svg-wrap"[^>]*>(.*?)</div>', html, re.DOTALL
    )
    assert tradeoff_svg is not None
    assert ">Reference<" in tradeoff_svg.group(1)
    assert ">Max Est. Tput<" in tradeoff_svg.group(1)
    assert ">Min Bound TPOT<" not in tradeoff_svg.group(1)
    assert ">Max Ceiling Tput<" not in tradeoff_svg.group(1)

    view_note = re.search(r'<p id="tradeoff-view-note"[^>]*>(.*?)</p>', html, re.DOTALL)
    assert view_note is not None
    assert (
        "3 embedded layouts (2 distinct plot positions; "
        "layouts with identical coordinates overlap)."
    ) in view_note.group(1)

    rows = _embedded_report_payload(html)["explorer"]["rows"]
    bound_svg = report_module._render_tradeoff_svg(
        list(enumerate(rows)), selected_idx=0, mode="bound", has_slo=False
    )
    assert ">Reference<" in bound_svg
    assert ">Min Bound TPOT<" in bound_svg
    assert ">Max Ceiling Tput<" in bound_svg
    assert ">Max Est. Tput<" not in bound_svg


def test_limits_chart_serial_op_overhead_family_overflow_and_single_host_hbm(
    sample_payload: SolvePayload,
) -> None:
    """Limits chart, mesh family members, and single-host HBM use unified labels."""
    payload = copy.deepcopy(dict(sample_payload))
    payload["single_host_fit"] = {
        "is_legal": True,
        "status": "LEGAL",
        "fit_reason": "Fits on 1 host",
        "hbm_used_gb": 61.56,
        "hbm_limit_gb": 103.0,
        "best_parallelism": "TP8",
        "topology": "2x2",
        "chips_per_host": 4,
        "cores_per_host": 8,
    }
    default_conc_key = str(
        max(int(cand["concurrency"]) for cand in payload["candidates"])
    )
    payload["bound_families_by_concurrency"] = {
        default_conc_key: [
            {
                "rank": 1,
                "family_label": "TP8",
                "min_bound_tpot_ms": 0.27,
                "min_bound_ttft_ms": 12.5,
                "min_envelope_tpot_ms": 0.27,
                "min_envelope_ttft_ms": 12.5,
                "binding": "hbm",
                "member_count": 3,
                "members": ["TP8 DP1", "TP8 DP1 EP8", "TP8 DP1 PP1"],
            },
            {
                "rank": 2,
                "family_label": "TP4",
                "min_bound_tpot_ms": 0.54,
                "min_bound_ttft_ms": 20.0,
                "min_envelope_tpot_ms": 0.54,
                "min_envelope_ttft_ms": 20.0,
                "binding": "hbm",
                "member_count": 2,
                "members": ["TP4 DP2", "TP4 DP2 EP8"],
            },
            {
                "rank": 3,
                "family_label": "TP2",
                "min_bound_tpot_ms": 1.0,
                "min_bound_ttft_ms": 30.0,
                "min_envelope_tpot_ms": 1.0,
                "min_envelope_ttft_ms": 30.0,
                "binding": "hbm",
                "member_count": 0,
                "members": [],
            },
        ]
    }
    html = render_html(payload)
    assert "Serial Op Overhead" in html
    assert ">Op Chain<" not in html
    assert '["Op Chain"' not in html
    assert (
        "The table's <code>Binding</code> column reflects the operating batch" in html
    )
    assert "<td>3 (TP8 DP1, TP8 DP1 EP8, +1 more)</td>" in html
    assert "<td>2 (TP4 DP2, TP4 DP2 EP8)</td>" in html
    assert "<td>0</td>" in html
    assert "HBM Footprint: <code>61.6 GB</code> / <code>103.0 GB</code> limit" in html


def test_report_keeps_exact_leaders_of_every_sort_when_rows_are_capped(
    sample_payload: SolvePayload,
) -> None:
    """A capped explorer still embeds the true first 15 layouts of each sort key."""
    num_layouts, leaders = 200, 15
    base = copy.deepcopy(dict(sample_payload["frontier"][0]))
    variants = []
    for idx in range(num_layouts):
        point = copy.deepcopy(base)
        point["concurrency"] = 64
        point["slo_verdict"] = "estimate_meets"
        point["decode_pool"]["kv_split"] = idx + 1
        point["bound_tpot_ms"] = 0.2 + ((idx * 53) % num_layouts) / 1000.0
        point["floor_throughput_per_chip"] = 5000.0 + ((idx * 29) % num_layouts) * 10
        point["token_throughput_per_chip"] = 1000.0 + ((idx * 37) % num_layouts)
        point["est_interactivity_tok_s"] = 10.0 + ((idx * 91) % num_layouts)
        point["decode_pool"]["hbm_used_gb"] = 40.0 + ((idx * 71) % num_layouts) / 10
        variants.append(point)
    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    payload["candidates"] = variants
    payload["frontier"] = []
    payload["top_k"] = []
    payload["top_k_limit"] = None
    payload["recommended"] = variants[0]
    reference_point = max(variants, key=lambda point: point["bound_tpot_ms"])
    payload["reference"] = {"found": True, "point": reference_point}
    payload.pop("concurrency_top_k", None)
    payload.pop("bound_families_by_concurrency", None)
    assert sample_payload["search"] is not None
    payload["search"] = {
        **dict(sample_payload["search"]),
        "core_grid": num_layouts,
        "enumerated": num_layouts,
        "hbm_fit": num_layouts,
        "not_ruled_out": num_layouts,
    }

    html = render_html(payload)
    rows = _embedded_report_payload(html)["explorer"]["rows"]
    assert len(rows) < num_layouts
    assert f"({len(rows):,} of {num_layouts:,})</button>" in html
    assert f"up to {len(rows)} per concurrency" in html
    assert 'class="pill-btn">Show All (' in html
    # The static chart plots the embedded rows, the same set the script redraws.
    chart = html[html.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert chart.count('style="cursor:pointer"') == len(rows)
    sort_fields = (
        ("bt", "bound_tpot_ms", False, 3),
        ("fc", "floor_throughput_per_chip", True, 1),
        ("et", "token_throughput_per_chip", True, 1),
        ("ei", "est_interactivity_tok_s", True, 1),
    )
    for row_field, point_field, descending, digits in sort_fields:
        true_leaders = sorted(
            (round(float(point[point_field]), digits) for point in variants),
            reverse=descending,
        )[:leaders]
        embedded_values = {row[row_field] for row in rows}
        assert set(true_leaders) <= embedded_values
    lowest_hbm = sorted(
        round(float(point["decode_pool"]["hbm_used_gb"]), 2) for point in variants
    )[:leaders]
    assert set(lowest_hbm) <= {row["hbm"] for row in rows}
    shown_bounds = [
        re.sub(r"<[^>]+>", "", re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)[4])
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", _explorer_tbody(html), re.DOTALL)
    ]
    # Bounds 0.200 to 0.214 ms: the cell prints the 3 decimals the row sorts by.
    assert shown_bounds[:leaders] == [f"0.2{step:02d}" for step in range(leaders)]

    # The reference has the highest bound, so it sits last in the embedded rows.
    # The first paint appends it after the leaders and keeps its row number.
    static_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", _explorer_tbody(html), re.DOTALL)
    assert len(static_rows) == leaders + 1
    assert [row["ref"] for row in rows].index(True) == len(rows) - 1
    reference_cells = re.findall(r"<td[^>]*>(.*?)</td>", static_rows[-1], re.DOTALL)
    assert reference_cells[0] == str(len(rows))
    assert "REF" in static_rows[-1]
    assert all("REF" not in row for row in static_rows[:-1])


def test_report_shows_no_slo_badge_unless_an_slo_is_set(
    sample_payload: SolvePayload,
) -> None:
    """Rows carry a verdict badge only when the scenario sets a TTFT or TPOT SLO."""
    tbody_no_slo = _explorer_tbody(render_html(sample_payload))
    assert "No SLO" in tbody_no_slo
    assert "Est. Meets" not in tbody_no_slo

    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    payload["scenario"] = {**dict(sample_payload["scenario"]), "tpot_slo_ms": 1.0e6}
    tbody_slo = _explorer_tbody(render_html(payload))
    assert "Est. Meets" in tbody_slo
    assert "No SLO" not in tbody_slo


@pytest.mark.parametrize(
    ("est_interactivity", "est_throughput"),
    [
        (151.8, 89449.8),
        (299.1, 25205.2),
        (852.7, 7804.2),
        (455.2, 6311.4),
        (0.26, 0.011),
        (0.012, 0.0004),
        (7.9, 21.0),
    ],
)
def test_tradeoff_points_stay_inside_plot_frame(
    sample_payload: SolvePayload,
    est_interactivity: float,
    est_throughput: float,
) -> None:
    """Every plotted layout sits inside the axis frame for any data maximum."""
    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    point = copy.deepcopy(dict(sample_payload["frontier"][0]))
    point["concurrency"] = 64
    point["est_interactivity_tok_s"] = est_interactivity
    point["token_throughput_per_chip"] = est_throughput
    payload["candidates"] = [point]
    payload["frontier"] = [point]
    payload["top_k"] = []
    payload["recommended"] = point
    html = render_html(payload)
    chart = html[html.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    _assert_plotted_points_inside_frame(chart)
    _assert_axis_ticks_distinct_and_evenly_spaced(chart)


def test_report_header_and_payload_hide_config_directories(
    sample_payload: SolvePayload,
) -> None:
    """A config path passed as the model shows as its file name only."""
    payload: dict[str, object] = dict(sample_payload)
    payload["model"] = "/home/someone/configs/qwen3_5_397b_fp8.json"
    html = render_html(payload)
    assert "<strong>Model:</strong> qwen3_5_397b_fp8.json" in html
    assert "/home/someone" not in html


def test_report_without_feasible_layout(sample_payload: SolvePayload) -> None:
    """When no layout is feasible, renders warning banner and empty state gracefully."""
    payload: dict[str, object] = dict(sample_payload)
    payload["recommended"] = None
    payload["top_k"] = []
    payload["warnings"] = ["SLO_UNMET: no feasible topology met latency constraints"]
    html = render_html(payload)
    assert "SLO_UNMET" in html
    assert 'xmlns="http://www.w3.org/2000/svg"' in html

    oom_payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    oom_payload["recommended"] = None
    oom_payload["candidates"] = []
    oom_payload["frontier"] = []
    oom_payload["top_k"] = []
    oom_payload["warnings"] = []
    oom_payload["search"] = {
        "enumerated": 10,
        "hbm_fit": 0,
        "slo_met": 0,
        "frontier": 0,
    }
    oom_html = render_html(oom_payload)
    assert "HBM memory constraints" in oom_html
    assert "recommended layout is infeasible" in oom_html


def test_report_embeds_explorer_rows_only_and_strips_family_members(
    sample_payload: SolvePayload,
) -> None:
    """The page embeds explorer rows, not raw candidate lists or family members."""
    payload: dict[str, object] = dict(sample_payload)
    cands = sample_payload["candidates"]
    base_cand = cands[0] if cands else {}
    payload["candidates"] = [dict(base_cand) for _ in range(3000)]
    payload["bound_families_by_concurrency"] = {
        "128": [
            {
                "rank": idx,
                "family": ("mixed", "2x2x1", f"TP{idx}"),
                "min_bound_tpot_ms": 1.25,
                "min_bound_ttft_ms": 42.0,
                "binding": "hbm",
                "member_count": 20,
                "members": [dict(base_cand) for _ in range(20)],
            }
            for idx in range(1, 51)
        ]
    }
    html = render_html(payload)
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    raw_json = match.group(1).replace(r"<\/", "</")

    def _reject_constant(val: str) -> object:
        msg = f"Unexpected non-finite JSON constant: {val}"
        raise AssertionError(msg)

    embedded = json.loads(raw_json, parse_constant=_reject_constant)
    assert "candidates" not in embedded
    assert "top_k" not in embedded
    # 3,000 copies of one layout embed the same rows as a single copy.
    single_copy = dict(payload)
    single_copy["candidates"] = [dict(base_cand)]
    rows_single_copy = _embedded_report_payload(render_html(single_copy))["explorer"][
        "rows"
    ]
    assert len(embedded["explorer"]["rows"]) == len(rows_single_copy)
    fams = embedded["bound_families_by_concurrency"]["128"]
    assert all("members" not in fam for fam in fams)
    assert all("sample_members" in fam for fam in fams)


def test_report_slo_unmet_and_undecided_verdict_rendering(
    sample_payload: SolvePayload,
) -> None:
    """Verify INFEASIBLE banner for ruled_out and non-infeasible badge for undecided."""
    html_ok = render_html(sample_payload)
    assert "<th>SLO Verdict</th>" in html_ok
    assert "<th>Tput Ceiling (tok/s/chip) [bound]</th>" in html_ok
    assert "<th>KV Cap / Prefix Hit Rate</th>" in html_ok
    assert "  renderAll();\n})();" in html_ok

    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    assert sample_payload["recommended"] is not None
    rec_unmet = dict(sample_payload["recommended"])
    rec_unmet["slo_met"] = False
    rec_unmet["slo_verdict"] = SLO_VERDICT_RULED_OUT
    payload["recommended"] = rec_unmet
    payload["top_k"] = []
    payload["warnings"] = []
    html_unmet = render_html(payload)
    assert "INFEASIBLE" in html_unmet
    assert '<div class="alert-box alert-warning">' in html_unmet
    assert "recommended layout is infeasible" in html_unmet

    payload_undecided: dict[str, object] = copy.deepcopy(dict(sample_payload))
    rec_undec = dict(sample_payload["recommended"])
    rec_undec["slo_met"] = False
    rec_undec["slo_verdict"] = "undecided"
    payload_undecided["recommended"] = rec_undec
    payload_undecided["warnings"] = []
    html_undec = render_html(payload_undecided)
    assert "UNDECIDED" in html_undec
    assert "recommended layout is infeasible" not in html_undec


def test_render_html_clean_svg_full_hbm_and_dynamic_chip_spec(
    sample_payload: SolvePayload,
) -> None:
    """Verify clean SVG formatting, full physical HBM spec, and dynamic chip_spec."""
    payload = copy.deepcopy(dict(sample_payload))
    rec = payload.get("recommended")
    if isinstance(rec, dict) and isinstance(rec.get("decode_pool"), dict):
        rec["decode_pool"].pop("hbm_limit_gb", None)
    html = render_html(payload)
    assert "</text>'" not in html
    assert "103.00 GB" in html
    assert "94.76" not in html
    assert "Assumptions, Hardware Specs" in html

    custom_chip = dc_replace(
        CHIPS["tpu7x"],
        name="custom_tpu_v6",
        hbm_bytes=32_000_000_000,
        chips_per_host=1,
        fp4_flops=4.6e15,
        int8_flops=2.3e15,
    )
    custom_payload = copy.deepcopy(dict(sample_payload))
    custom_payload["chip"] = "custom_tpu_v6"
    custom_payload["chip_spec"] = {
        "name": custom_chip.name,
        "hbm_bytes": custom_chip.hbm_bytes,
        "hbm_bw": custom_chip.hbm_bw,
        "bf16_flops": custom_chip.bf16_flops,
        "fp8_flops": custom_chip.fp8_flops,
        "fp4_flops": custom_chip.fp4_flops,
        "int8_flops": custom_chip.int8_flops,
        "vector_flops": custom_chip.vector_flops,
        "chips_per_host": 1,
        "vmem_bytes": custom_chip.vmem_bytes,
        "ici_link_bw": custom_chip.ici_link_bw,
        "ici_bw": custom_chip.ici_bw,
        "ici_hop_latency_s": custom_chip.ici_hop_latency_s,
    }
    custom_html = render_html(custom_payload)
    assert "custom_tpu_v6 Per-Core Specs:" in custom_html
    assert "32.0 GB" in custom_html
    assert "FP4" in custom_html
    assert "INT8" in custom_html
    assert "Vector" in custom_html
    assert "1 chip / 1 host" in custom_html


def test_report_uses_explicit_concurrency_hit_rate_and_chip_headers(
    sample_payload: SolvePayload,
) -> None:
    """Verify HTML report spells out Concurrency, Prefix Hit Rate, and chips headers."""
    html = render_html(sample_payload)
    assert "Concurrency" in html
    assert "Prefix Hit Rate" in html
    assert "chips" in html
    assert not re.search(r"<th[^>]*>\s*c\s*</th>", html, re.IGNORECASE)
    assert not re.search(r"<th[^>]*>\s*h\s*</th>", html, re.IGNORECASE)
    assert not re.search(r"\d+c / \d+h", html)


def test_frontier_chart_plots_per_concurrency_envelope_without_label_overlap(
    sample_payload: SolvePayload,
) -> None:
    """Per-concurrency chart keeps undecided points with non-overlapping labels."""
    payload = copy.deepcopy(sample_payload)
    base = dict(payload["frontier"][0])
    p_low = copy.deepcopy(base)
    p_low["concurrency"] = 64
    p_low["est_interactivity_tok_s"] = 1000.0
    p_low["token_throughput_per_chip"] = 2500.0
    p_low["decode_pool"]["tp"] = 8
    p_low["decode_pool"]["dp"] = 1
    p_high = copy.deepcopy(base)
    p_high["concurrency"] = 64
    p_high["est_interactivity_tok_s"] = 1000.0
    p_high["token_throughput_per_chip"] = 10000.0
    p_high["decode_pool"]["tp"] = 4
    p_high["decode_pool"]["dp"] = 2
    p_tput = copy.deepcopy(base)
    p_tput["concurrency"] = 64
    p_tput["est_interactivity_tok_s"] = 200.0
    p_tput["token_throughput_per_chip"] = 40000.0
    p_tput["slo_verdict"] = "undecided"
    p_tput["decode_pool"]["tp"] = 2
    p_tput["decode_pool"]["dp"] = 4
    payload["frontier"] = (p_low, p_high, p_tput)
    payload["candidates"] = (p_low, p_high, p_tput)
    payload["top_k"] = []
    payload["recommended"] = p_high

    html = render_html(payload)
    chart = html[html.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert "Est. Interactivity (tok/s)</text>" in chart
    assert "Est. Token Throughput per Chip (tok/s/chip)</text>" in chart
    assert "TPOT bound=" in chart
    match = re.search(r'<polyline points="([^"]+)"', chart)
    assert match is not None
    coords = [
        tuple(float(v) for v in pair.split(",")) for pair in match.group(1).split()
    ]
    assert len(coords) == 2
    assert coords[0][0] < coords[1][0]
    assert coords[0][1] < coords[1][1]

    # Verify placed label pill bounding boxes do not overlap.
    rects = [
        (float(x), float(y), float(x) + float(w), float(y) + float(h))
        for x, y, w, h in re.findall(
            r'<rect x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" '
            r'height="([0-9.]+)" rx="4" fill="#ffffff"',
            chart,
        )
    ]
    for i, (x0, y0, x1, y1) in enumerate(rects):
        for ox0, oy0, ox1, oy1 in rects[i + 1 :]:
            assert not (x0 < ox1 and x1 > ox0 and y0 < oy1 and y1 > oy0)


def test_report_renders_four_stage_physics_funnel_and_submodule_pools(
    sample_payload: SolvePayload,
) -> None:
    """HTML report renders 4-stage physics funnel, Attn | MoE pools, no shortlist."""
    moe_payload = copy.deepcopy(sample_payload)
    moe_payload["has_moe"] = True
    assert moe_payload["search"] is not None
    moe_search = dict(moe_payload["search"])
    moe_search["core_grid"] = 2940
    moe_search["enumerated"] = 98
    moe_search["hbm_fit"] = 98
    moe_search["not_ruled_out"] = 98
    moe_payload["search"] = moe_search
    moe_payload["budget"] = 4
    moe_payload["chip_budget"] = {"max_chips": 4, "budget_source": "flag"}
    rank1_point = copy.deepcopy(moe_payload["top_k"][0]["point"])
    rank1_point["slo_verdict"] = "estimate_meets"
    rank1_point["concurrency"] = 64
    undecided_point = copy.deepcopy(rank1_point)
    undecided_point["slo_verdict"] = "undecided"
    undecided_point["decode_pool"]["kv_split"] = 2
    rank1_tput = rank1_point["token_throughput_per_chip"]
    assert isinstance(rank1_tput, (int, float))
    undecided_point["token_throughput_per_chip"] = float(rank1_tput) * 1.15
    moe_payload["top_k"][0]["point"] = rank1_point
    moe_payload["frontier"] = (rank1_point, undecided_point)
    moe_payload["candidates"] = (rank1_point, undecided_point)

    html = render_html(moe_payload)
    funnel = html[html.index('id="search-funnel"') :]
    funnel = funnel[: funnel.index("</section>")]
    assert "Core-Valid Grid" in funnel
    assert "2,940" in funnel
    assert "Physically Legal" in funnel
    assert "HBM Fit" in funnel
    assert "Est. 2D Frontier" not in funnel
    assert len(re.findall(r'class="funnel-step', funnel)) == 4
    assert "Architecture &amp; Runtime Legal" not in funnel
    assert "--max-num-seqs" not in funnel
    assert "Attn:" in html
    assert "MoE:" in html
    assert "1 hosts" not in html
    assert 'data-conc-btn="64"' in html


def test_report_honest_analytical_headings_and_excludes_ruled_out_envelope(
    sample_payload: SolvePayload,
) -> None:
    """Verify honest Layer-1 headings and ruled-out exclusion from upper envelope."""
    html_base = render_html(sample_payload)
    assert "Physical Envelope &amp; Bounds Summary (Layer 1)" in html_base
    assert "One decode step with the fewest resident sequences" in html_base
    assert "shortlist" not in html_base.lower()
    assert "Executive Recommendation" not in html_base
    assert "Recommended Launch Command" not in html_base
    assert "Best Parallelism:" not in html_base
    assert 'id="bound-families-table"' in html_base
    no_families_payload = dict(sample_payload)
    no_families_payload.pop("bound_families_by_concurrency", None)
    assert 'id="bound-families-table"' not in render_html(no_families_payload)

    payload = copy.deepcopy(sample_payload)
    ruled_out_pts = []
    for point in payload["candidates"]:
        copy_pt = dict(point)
        copy_pt["slo_verdict"] = "ruled_out"
        copy_pt["slo_met"] = False
        ruled_out_pts.append(copy_pt)
    payload["candidates"] = tuple(ruled_out_pts)
    payload["frontier"] = tuple(ruled_out_pts)
    payload["top_k"] = []
    payload["recommended"] = ruled_out_pts[0] if ruled_out_pts else None
    html_ruled = render_html(payload)
    chart = html_ruled[html_ruled.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert "<polyline" not in chart


def test_report_renders_optional_reserve_provenance_assumptions_and_families(
    sample_payload: SolvePayload,
) -> None:
    """Verify guarded rendering when HBM reserve, provenance, and families exist."""
    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    reserve_gb = 12.0
    expected_usable = 103.0 - reserve_gb
    payload["hbm_reserve_gb"] = reserve_gb
    payload["single_host_fit"] = {
        "feasible": True,
        "status": "FEASIBLE",
        "reason": "Fits within reserved HBM",
        "hbm_used_gb": 50.0,
        "hbm_limit_gb": expected_usable,
        "best_parallelism": "TP8",
    }
    payload["provenance"] = {
        "calibrated": False,
        "basis": "Analytical bounds and uncalibrated serial estimates",
        "ranking_basis": "bound_ordered_handoff",
        "supplied_values": {
            "hbm_reserve_gb": {
                "flag": "--hbm-reserve-gb",
                "value": reserve_gb,
                "kind": "assumed",
                "unit": "GB",
            },
            "kv_dtype": {
                "flag": "--kv-dtype",
                "value": "fp8",
                "kind": "config",
                "unit": "dtype",
            },
        },
    }
    payload["solver_assumptions"] = [
        {"name": "moe_activation", "value": "dense_fallback", "kind": "derived"},
    ]
    sample_point = copy.deepcopy(sample_payload["frontier"][0])
    payload["bound_families_by_concurrency"] = {
        "16": [
            {
                "rank": 1,
                "family": ("mixed", "2x2x1", "TP8"),
                "min_bound_tpot_ms": 1.25,
                "min_bound_ttft_ms": 42.0,
                "min_envelope_tpot_ms": 1.45,
                "min_envelope_ttft_ms": 48.5,
                "binding": "hbm",
                "any_not_ruled_out": True,
                "member_count": 2,
                "members": [sample_point],
            }
        ]
    }
    html = render_html(payload)
    assert f"{expected_usable:.2f} GB" in html
    assert f"minus <code>{reserve_gb:.2f} GB</code> HBM reserve" in html
    assert "HBM Workspace Reserve:" in html
    assert "HBM LEGAL (NOT RULED OUT)" in html
    assert "--hbm-reserve-gb=12.0</code> GB" in html
    assert "--kv-dtype=fp8</code>" in html
    assert "--kv-dtype=fp8</code> dtype" not in html
    assert "bound_ordered_handoff" in html
    assert "Solver Assumptions:" in html
    assert "moe_activation=dense_fallback" in html
    assert 'id="bound-families-table"' in html
    assert "mixed | 2x2x1 | TP8" in html
    assert "1.25" in html


def test_report_single_host_fit_renders_na_when_best_parallelism_is_none(
    sample_payload: SolvePayload,
) -> None:
    """Render N/A rather than None when single-host fit has no valid layout."""
    payload = copy.deepcopy(sample_payload)
    payload["single_host_fit"] = {
        "feasible": False,
        "status": "REJECTED",
        "reason": "Exceeds single-host HBM",
        "hbm_used_gb": 120.0,
        "hbm_limit_gb": 103.0,
        "best_parallelism": None,
    }
    html = render_html(payload)
    assert "Sample Single-Host Layout: <code>N/A</code>" in html
    assert "<code>None</code>" not in html


def test_report_renders_infeasible_banner_with_existing_warnings(
    sample_payload: SolvePayload,
) -> None:
    """Infeasible banner renders alongside pre-existing informational warnings."""
    payload_ruled: dict[str, object] = copy.deepcopy(dict(sample_payload))
    raw_rec = payload_ruled["recommended"]
    assert isinstance(raw_rec, dict)
    record = dict(raw_rec)
    record["slo_verdict"] = SLO_VERDICT_RULED_OUT
    record["slo_met"] = False
    payload_ruled["recommended"] = record
    payload_ruled["warnings"] = [UNKNOWN_THINK_TIME_WARNING]

    html_ruled = render_html(payload_ruled)
    assert UNKNOWN_THINK_TIME_WARNING in html_ruled
    assert "recommended layout is infeasible" in html_ruled

    payload_none: dict[str, object] = copy.deepcopy(dict(sample_payload))
    payload_none["recommended"] = None
    payload_none["top_k"] = []
    payload_none["warnings"] = [UNKNOWN_THINK_TIME_WARNING]

    html_none = render_html(payload_none)
    assert UNKNOWN_THINK_TIME_WARNING in html_none
    assert "recommended layout is infeasible" in html_none


def test_report_renders_na_instead_of_zero_chips_one_host(
    sample_payload: SolvePayload,
) -> None:
    """Empty surviving set renders N/A instead of 0 chips (1 host)."""
    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    payload["recommended"] = None
    payload["candidates"] = []
    payload["frontier"] = []
    payload["top_k"] = []
    payload["concurrency_top_k"] = {}
    payload["bound_families_by_concurrency"] = {}
    html = render_html(payload)
    assert "0 chips (1 host)" not in html
    assert "Concurrency 0 across" not in html
    assert "&bull; N/A" in html


@pytest.mark.parametrize("fixture_name", LOCKED_FIXTURE_FILES)
@pytest.mark.parametrize("accept_len", (1.0, 3.0))
def test_report_bounds_limits_bars_and_accept_len_across_locked_fixtures(
    fixture_name: str,
    accept_len: float,
) -> None:
    """Verify positive TPOT bounds <= est TPOT and split limits bars on all models."""
    bundle = load_locked_bundle(fixture_name)
    chip = CHIPS["tpu7x"]
    scenario = TrafficScenario(
        name="report-bounds-check",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=accept_len,
        think_time_s=0.0,
        concurrencies=(1, 8),
        kv_working_set_turns=1.0,
        spec_tokens=3 if accept_len > 1.0 else None,
    )
    plan = solve_serving_plan(
        bundle.geometry,
        bundle.dtypes,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    payload = build_solve_payload(
        plan,
        chip=chip,
        dtypes=bundle.dtypes,
        model=bundle.identity.hf_repo,
        top_k=None,
    )
    html = render_html(payload)
    explorer_sec = html[html.index('id="candidate-explorer"') :]
    explorer_sec = explorer_sec[: explorer_sec.index("</section>")]
    tbody_match = re.search(
        r'<tbody id="explorer-tbody">(.*?)</tbody>', explorer_sec, re.DOTALL
    )
    assert tbody_match is not None
    tr_blocks = re.findall(r"<tr[^>]*>(.*?)</tr>", tbody_match.group(1), re.DOTALL)
    assert tr_blocks
    for tr_html in tr_blocks:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr_html, re.DOTALL)
        tpot_bound = float(re.sub(r"<[^>]+>", "", cells[4]).replace(",", ""))
        tpot_est = float(re.sub(r"<[^>]+>", "", cells[9]).replace(",", ""))
        assert 0.0 < tpot_bound <= tpot_est + 1e-6

    limits_sec = html[html.index('id="limits-chart"') :]
    limits_sec = limits_sec[: limits_sec.index("</section>")]
    assert "PHYSICAL LOWER BOUNDS [bound_*]" in limits_sec
    assert "SERIAL ESTIMATE OVERHEAD [est_*]" in limits_sec
    bar_vals = {
        label: (float(w_str), float(ms_str))
        for w_str, label, ms_str in re.findall(
            r'width="([0-9.]+)"[^>]*><title>([^:]+): ([0-9.]+) ms</title>',
            limits_sec,
        )
    }
    assert "Hard Floor" in bar_vals
    hard_ms = bar_vals["Hard Floor"][1]
    for bound_name in ("HBM Bound", "MatMul Bound", "Vector Bound", "Collective Bound"):
        width_px, val_ms = bar_vals[bound_name]
        assert val_ms <= hard_ms + 1e-3
        if val_ms == 0.0:
            assert width_px == 0.0

    frontier_sec = html[html.index('id="frontier-chart"') :]
    frontier_sec = frontier_sec[: frontier_sec.index("</section>")]
    ring_count = len(re.findall(r'<circle [^>]*r="9\.5" fill="none"', frontier_sec))
    assert ring_count <= 4
    _assert_plotted_points_inside_frame(frontier_sec)


def test_report_pins_disagg_edge_guards_and_cli_html_output(
    tmp_path: Path,
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify P:D pin display, dtype formatting, edge guards, write_html, and CLI."""
    cfg_path = locked_config_path("qwen3_5_397b_nvfp4.json")
    out_html = tmp_path / "out_report.html"
    rc = cli.main([*_disagg_solve_argv(cfg_path), "--html", str(out_html), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert out_html.is_file()
    html_text = out_html.read_text(encoding="utf-8")
    assert "P:D=1:1" in html_text
    assert "<th>Prefill Pool</th>" in html_text
    assert "<th>Decode Pool</th>" in html_text
    assert 'id="single-host-fit"' not in html_text
    assert str(cfg_path.parent) not in html_text
    assert "<th>TPOT Bound (ms) [bound]</th>" in html_text
    pool_tables = html_text[html_text.index('id="pool-tables"') :]
    pool_tables = pool_tables[: pool_tables.index('id="limits-chart"')]
    assert "<th>TPOT Floor (ms)</th><th>TTFT Floor (ms)</th>" in pool_tables
    assert "<th>TPOT Bound" not in pool_tables
    assert "TPOT Floor (Decode Pools tab)" in html_text
    assert "TTFT floor is the same at every concurrency" in pool_tables
    assert "act=a4" in html_text
    assert "state=float32" in html_text

    mutated = copy.deepcopy(payload)
    mutated["has_moe"] = False
    mutated["dtypes"]["w_expert"] = MXFP4_BYTES
    mutated["dtypes"]["w_expert_dtype"] = "mxfp4"
    for cand in mutated["candidates"]:
        cand["effective_prefix_hit_rate"] = None
    for cand in mutated["frontier"]:
        cand["effective_prefix_hit_rate"] = None
    for entry in mutated["top_k"]:
        entry["point"]["effective_prefix_hit_rate"] = None
    mutated["recommended"]["effective_prefix_hit_rate"] = None
    mutated["recommended"]["decode_pool"]["hbm_limit_gb"] = 0.0
    mutated["recommended"]["decode_pool"]["hbm_used_gb"] = 150.0
    mutated["hbm_reserve_gb"] = 1000.0
    dest_path = write_html(mutated, tmp_path / "edge.html")
    edge_html = dest_path.read_text(encoding="utf-8")
    assert f"{MXFP4_BYTES:.6g}B" in edge_html
    assert " / -</td>" in edge_html
    assert "MoE pools show Attention axes" not in edge_html


def test_pool_tabs_list_each_concurrency_by_floor(
    locked_config_path: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pool tab keeps each concurrency's 15 lowest-floor pools, lowest first."""
    cfg_path = locked_config_path("qwen3_5_397b_nvfp4.json")
    assert cli.main([*_disagg_solve_argv(cfg_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    payload["scenario"]["tpot_slo_ms"] = 0.5
    template = payload["decode_pools"][0]

    def _decode_pool(
        concurrency: int,
        floor_ms: float | None,
        throughput: float,
        *,
        slo_met: bool = True,
    ) -> dict[str, object]:
        record = copy.deepcopy(template)
        record["concurrency"] = concurrency
        record["tpot_floor_ms"] = floor_ms
        record["token_throughput_per_chip"] = throughput
        record["slo_met"] = slo_met
        return record

    pools_per_concurrency, kept = 40, 15
    regular_floors = {
        concurrency: [
            concurrency + ((idx * 17) % pools_per_concurrency) / 10
            for idx in range(pools_per_concurrency)
        ]
        for concurrency in (2, 4)
    }
    records = [
        _decode_pool(concurrency, floor_ms, 1000.0 + idx)
        for concurrency, floors in regular_floors.items()
        for idx, floor_ms in enumerate(floors)
    ]
    # At c=4: two pools tie on the lowest floor, one pool's floor is above the SLO
    # (ruled out), and one pool has no floor.
    records += [
        _decode_pool(4, 3.5, 10.0),
        _decode_pool(4, 3.5, 5000.0),
        _decode_pool(4, 1.0, 9000.0, slo_met=False),
        _decode_pool(4, None, 9000.0),
    ]
    payload["decode_pools"] = records

    html = render_html(payload)
    assert f"Decode Pools ({2 * kept} of {len(records)})</button>" in html
    embedded = _embedded_report_payload(html)["explorer"]["decode_pools"]
    at_c4 = [(row["bt"], row["et"]) for row in embedded if row["c"] == 4]
    # The tie goes to the higher throughput estimate; the ruled-out pool and the
    # pool without a floor sort after every other pool and fall outside the 15.
    assert at_c4[:2] == [(3.5, 5000.0), (3.5, 10.0)]
    assert [floor for floor, _ in at_c4[2:]] == sorted(regular_floors[4])[: kept - 2]
    assert [row["bt"] for row in embedded if row["c"] == 2] == sorted(
        regular_floors[2]
    )[:kept]

    # The first paint lists the default concurrency (4, the solved one) only,
    # like the script's redraw, not the first rows of the lower concurrency.
    static_body = re.search(
        r'<tbody id="decode-pools-tbody">(.*?)</tbody>', html, re.DOTALL
    )
    assert static_body is not None
    assert static_body.group(1).count("<code>c=4</code>") == kept
    assert "<code>c=2</code>" not in static_body.group(1)


def _one_member_family(member: dict[str, object]) -> dict[str, object]:
    """Return a serialized mesh family holding one member layout."""
    return {
        "rank": 1,
        "family": ("mixed", "2x2x1", "TP8"),
        "min_bound_tpot_ms": 1.25,
        "min_bound_ttft_ms": 42.0,
        "binding": "hbm",
        "member_count": 1,
        "members": [member],
    }


def test_report_without_funnel_counts_names_the_payload_total_when_capped(
    sample_payload: SolvePayload,
) -> None:
    """Without funnel counters, counts are out of the layouts the payload carries."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 120)
    payload["search"] = None
    html = render_html(payload)
    assert _candidates_tab_count(html) == "80 of 120"
    assert (
        "Plots and lists 80 of 120 layout-concurrency points carried in the payload "
        "that are not ruled out (the report keeps the reference, the first 15 of "
        "each sort option"
    ) in html
    assert (
        "120 layout-concurrency points carried in the payload are not ruled out, "
        "across 1 concurrency"
    ) in html


def test_report_with_every_layout_ruled_out_says_none_passes(
    sample_payload: SolvePayload,
) -> None:
    """When the bound check rules out every layout, the note says so."""
    payload = _payload_with_verdicts(sample_payload, ["ruled_out"] * 3)
    payload["scenario"] = {**dict(sample_payload["scenario"]), "tpot_slo_ms": 0.01}
    assert sample_payload["search"] is not None
    payload["search"] = {
        **dict(sample_payload["search"]),
        "hbm_fit": 3,
        "slo_met": 0,
        "not_ruled_out": 0,
    }
    html = render_html(payload)
    assert _candidates_tab_count(html) == "0 not ruled out"
    assert (
        "No layout-concurrency point passes the bound check. "
        "The 3 ruled-out points are plotted and listed."
    ) in html
    assert "All 0" not in html

    # With nothing ruled out either, the note names no check as the cause.
    payload["candidates"] = []
    payload["recommended"] = None
    html = render_html(payload)
    assert "No layout-concurrency point is left to plot." in html
    assert "passes the bound check" not in html


def test_scope_note_names_reference_and_family_layouts_only_when_carried(
    sample_payload: SolvePayload,
) -> None:
    """The --top-k note lists the reference and family layouts only if present."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 2)
    payload["top_k_limit"] = 3
    assert sample_payload["search"] is not None
    payload["search"] = {
        **dict(sample_payload["search"]),
        "hbm_fit": 9,
        "not_ruled_out": 9,
    }
    carried = "the solver's top-3 list and recommended layout, "
    assert carried + "and the Pareto-marked layouts)." in render_html(payload)

    payload["reference"] = {"found": True, "point": payload["candidates"][1]}
    assert (
        carried + "the Pareto-marked layouts, and the reference layout)."
    ) in render_html(payload)

    payload["bound_families_by_concurrency"] = {
        "64": [_one_member_family(payload["candidates"][0])]
    }
    assert (
        carried + "the Pareto-marked layouts, the reference layout, and one layout "
        "per mesh family)."
    ) in render_html(payload)


def test_first_paint_selects_the_layout_the_script_selects(
    sample_payload: SolvePayload,
) -> None:
    """Meta lines, highlighted row, and highlighted point show the first row."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 2)
    first, second = payload["candidates"]
    first["bound_tpot_ms"], first["token_throughput_per_chip"] = 0.2001, 100.0
    second["bound_tpot_ms"], second["token_throughput_per_chip"] = 0.2004, 200.0
    first["decode_floor"]["hard_floor_ms"] = 0.333
    second["decode_floor"]["hard_floor_ms"] = 0.111
    html = render_html(payload)
    explorer = _embedded_report_payload(html)["explorer"]
    assert len({row["dm"] for row in explorer["rows"]}) == 2
    selected = explorer["rows"][explorer["default_selected_idx"]]
    # Both bounds show as 0.200 ms, so the higher throughput estimate goes first.
    assert selected["et"] == 200.0
    label = html_lib.escape(f"Selected: {selected['dm']} (c=64)")
    assert html.count(f">{label}</p>") == 2
    selected_row = re.search(
        r'<tr data-row-idx="\d+" class="row-selected">(.*?)</tr>',
        _explorer_tbody(html),
        re.DOTALL,
    )
    assert selected_row is not None
    assert f">{html_lib.escape(selected['dm'])}<" in selected_row.group(1)
    highlighted = re.findall(r'<circle [^>]*r="6.5"[^>]*><title>([^<]*)</title>', html)
    assert len(highlighted) == 1
    assert html_lib.unescape(highlighted[0]).startswith(f"{selected['dm']} | c=64 | ")
    # The step chart draws the selected row's floor, not the lowest raw bound's.
    limits_chart = html[html.index('id="limits-chart"') :]
    limits_chart = limits_chart[: limits_chart.index("</section>")]
    assert selected["df"][4] == 0.111
    assert "Hard Floor: 0.111 ms" in limits_chart
    assert "TPOT Bound = Hard Floor.</p>" in limits_chart
    assert "0.333 ms" not in limits_chart


def test_family_tab_first_paint_is_empty_when_default_concurrency_has_no_family(
    sample_payload: SolvePayload,
) -> None:
    """The static family table shows the default concurrency only, as the script."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 2)
    low_concurrency_point, high_concurrency_point = payload["candidates"]
    high_concurrency_point["concurrency"] = 128
    payload["bound_families_by_concurrency"] = {
        "64": [_one_member_family(low_concurrency_point)]
    }
    html = render_html(payload)
    assert _embedded_report_payload(html)["explorer"]["default_conc"] == 128
    family_body = re.search(
        r'<tbody id="families-tbody">(.*?)</tbody>', html, re.DOTALL
    )
    assert family_body is not None
    assert "No mesh families at this concurrency" in family_body.group(1)
    assert "<code>c=64</code>" not in family_body.group(1)


def test_pool_floor_glossary_appears_only_with_pool_tables(
    sample_payload: SolvePayload,
) -> None:
    """A report without pool tabs carries no pool-tab glossary entry."""
    payload: dict[str, object] = dict(sample_payload)
    payload.pop("prefill_pools", None)
    payload.pop("decode_pools", None)
    assert "Pools tab)" not in render_html(payload)


def _static_tradeoff_chart(html: str) -> str:
    """Return the static trade-off chart markup of a rendered report."""
    chart = html[html.index('id="tradeoff-svg-wrap"') :]
    return chart[: chart.index("</svg>")]


def test_tradeoff_callouts_follow_the_script_tie_rules(
    sample_payload: SolvePayload,
) -> None:
    """A tied callout goes to the first row in bound order; the reference is dashed."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 3)
    estimates = ((100.0, 10.0), (300.0, 20.0), (300.0, 30.0))
    for point, (throughput, interactivity) in zip(
        payload["candidates"], estimates, strict=True
    ):
        point["token_throughput_per_chip"] = throughput
        point["est_interactivity_tok_s"] = interactivity
    payload["reference"] = {"found": True, "point": payload["candidates"][2]}
    chart = _static_tradeoff_chart(render_html(payload))

    def _center(markup: str) -> tuple[str, str]:
        found = re.search(r'cx="([0-9.]+)" cy="([0-9.]+)"', markup)
        assert found is not None
        return found.group(1), found.group(2)

    points = {
        title.rsplit(", ", 1)[1]: _center(attrs)
        for attrs, title in re.findall(
            r'<circle ([^>]*style="cursor:pointer")><title>([^<]*)</title>', chart
        )
    }
    rings = {
        re.search(r'stroke="(#[0-9a-f]{6})"', attrs).group(1): attrs
        for attrs in re.findall(r'<circle ([^>]*r="9.5"[^>]*)/>', chart)
    }
    # Two layouts tie on the highest estimate; the earlier one (20 tok/s) is ringed.
    assert _center(rings["#d97706"]) == points["20.0 tok/s"]
    assert _center(rings["#7c3aed"]) == points["30.0 tok/s"]
    assert 'stroke-dasharray="3,2"' in rings["#7c3aed"]
    assert "stroke-dasharray" not in rings["#d97706"]


def test_tradeoff_envelope_skips_points_tied_on_throughput(
    sample_payload: SolvePayload,
) -> None:
    """The dashed envelope keeps a point only if it adds throughput."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 3)
    estimates = ((100.0, 30.0), (100.0, 20.0), (200.0, 10.0))
    for point, (throughput, interactivity) in zip(
        payload["candidates"], estimates, strict=True
    ):
        point["token_throughput_per_chip"] = throughput
        point["est_interactivity_tok_s"] = interactivity
    html = render_html(payload)
    envelope = re.search(r'<polyline points="([^"]*)"', _static_tradeoff_chart(html))
    assert envelope is not None
    # (20 tok/s, 100) sits under (30 tok/s, 100), so only 2 points remain.
    assert len(envelope.group(1).split()) == 2
    # The inline script uses the same tie tolerance as the static chart.
    assert f"var ENVELOPE_TIE_TOLERANCE = {FRONTIER_2D_TIE_TOLERANCE!r};" in html


def test_family_tab_first_paint_lists_concurrencies_ascending_without_a_default(
    sample_payload: SolvePayload,
) -> None:
    """With no default concurrency, family rows follow ascending concurrency."""
    payload: dict[str, object] = copy.deepcopy(dict(sample_payload))
    payload["candidates"] = []
    payload["frontier"] = []
    payload["top_k"] = []
    payload["recommended"] = None
    payload.pop("concurrency_top_k", None)
    family = {
        "rank": 1,
        "family": ("mixed", "2x2x1", "TP8"),
        "min_bound_tpot_ms": 1.25,
        "min_bound_ttft_ms": 42.0,
        "binding": "hbm",
        "member_count": 0,
        "members": [],
    }
    payload["bound_families_by_concurrency"] = {"128": [family], "64": [family]}
    html = render_html(payload)
    assert _embedded_report_payload(html)["explorer"]["default_conc"] is None
    family_body = re.search(
        r'<tbody id="families-tbody">(.*?)</tbody>', html, re.DOTALL
    )
    assert family_body is not None
    assert re.findall(r"<code>c=(\d+)</code>", family_body.group(1)) == ["64", "128"]


def test_number_formatting_follows_the_inline_script_rules() -> None:
    """Cell text rounds half-up on the shortest decimal; coordinates as toFixed."""
    # Cells and ticks: shortest decimal form, ties up, clamped to [0, 1e12].
    assert report_module._fixed(0.15, 1) == "0.2"
    assert report_module._fixed(0.125, 2) == "0.13"
    assert report_module._fixed(1234.25, 1, grouped=True) == "1,234.3"
    assert report_module._fixed(5e-7, 6) == "0.000001"
    assert report_module._fixed(-0.04, 1) == "0.0"
    assert report_module._fixed(2e12, 1) == "1000000000000.0"
    # Coordinates: exact binary value, so 0.15 (just under 0.15) rounds down.
    assert report_module._to_fixed(0.15, 1) == "0.1"
    assert report_module._to_fixed(0.125, 2) == "0.13"
    assert report_module._to_fixed(2.5, 0) == "3"


def test_reference_card_and_step_note_state_how_numbers_are_defined(
    sample_payload: SolvePayload,
) -> None:
    """The card gives the rank among layouts; the step note ties floor to bound."""
    payload = _payload_with_verdicts(sample_payload, ["estimate_meets"] * 2)
    payload["scenario"] = {**dict(sample_payload["scenario"]), "accept_len": 2.0}
    payload["reference"] = {
        "found": True,
        "point": payload["candidates"][1],
        "rank_by_tpot_bound": 1327,
        "candidates": 1734,
        "concurrency": 64,
    }
    html = render_html(payload)
    assert "TPOT bound rank 1,327 of 1,734 layouts at c=64 (ties share a rank)" in html
    assert "TPOT Bound = Hard Floor / 2 accepted tokens per step.</p>" in html
    assert "<th>Row</th>" in html
    assert "rows after the first 15 are not ranks" in html

    payload["reference"] = {"found": True, "point": payload["candidates"][1]}
    assert "Survived (L1 legal)" in render_html(payload)
