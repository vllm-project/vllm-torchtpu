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
"""CPU unit tests for roofline HTML report generation."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    SolveOptions,
    TrafficScenario,
    build_solve_payload,
    render_html,
    solve_serving_plan,
)
from vllm_torchtpu.roofline.emit import pool_label
from vllm_torchtpu.roofline.geometry import from_hf_config
from vllm_torchtpu.roofline.report import MAX_EMBEDDED_CANDIDATES

if TYPE_CHECKING:
    from vllm_torchtpu.roofline import SolvePayload


@pytest.fixture
def sample_payload() -> SolvePayload:
    """Return a valid solve payload for report testing."""
    cfg = {
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 8,
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 32000,
    }
    geom = from_hf_config(cfg)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(kv=1.0)
    scenario = TrafficScenario(
        name="chat-8k1k",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 2),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    return build_solve_payload(plan, chip=chip, dtypes=dt, model="test-llama", top_k=3)


def test_report_html_is_deterministic(sample_payload: SolvePayload) -> None:
    """render_html produces byte-identical output across consecutive calls."""
    html1 = render_html(sample_payload)
    html2 = render_html(sample_payload)
    min_html_len = 500
    assert html1 == html2
    assert len(html1) > min_html_len


def test_report_has_no_external_fetch(sample_payload: SolvePayload) -> None:
    """HTML report is fully offline: zero external URLs, stylesheets, or scripts."""
    html = render_html(sample_payload)
    assert "<link" not in html
    assert not re.search(r"<script[^>]+src=", html, re.IGNORECASE)
    assert "https://" not in html
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


def test_report_sections_present(sample_payload: SolvePayload) -> None:
    """All 7 required functional sections and anchors are present."""
    html = render_html(sample_payload)
    assert 'id="executive-summary"' in html
    assert 'id="search-funnel"' in html
    assert 'id="frontier-chart"' in html
    assert 'id="limits-chart"' in html
    assert 'id="memory-chart"' in html
    assert 'id="top-layouts-table"' in html
    assert 'id="roofline-data"' in html
    assert "<svg" in html


def test_report_without_feasible_layout(sample_payload: SolvePayload) -> None:
    """When no layout is feasible, renders warning banner and empty state gracefully."""
    payload: dict[str, object] = dict(sample_payload)
    payload["recommended"] = None
    payload["top_k"] = []
    payload["warnings"] = ["SLO_UNMET: no feasible topology met latency constraints"]
    html = render_html(payload)
    assert "SLO_UNMET" in html
    assert "SLO_UNMET: no feasible topology met latency constraints" in html
    assert "<svg" in html


def test_report_caps_embedded_candidates(sample_payload: SolvePayload) -> None:
    """Payloads with large candidate counts downsample embedded JSON."""
    expected_cap = 200
    payload: dict[str, object] = dict(sample_payload)
    cands = sample_payload["candidates"]
    base_cand = cands[0] if cands else {}
    payload["candidates"] = [dict(base_cand) for _ in range(3000)]
    html = render_html(payload)
    match = re.search(
        r'<script type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    raw_json = match.group(1).replace(r"<\/", "</")
    embedded = json.loads(raw_json)
    assert expected_cap == MAX_EMBEDDED_CANDIDATES
    assert len(embedded["candidates"]) == MAX_EMBEDDED_CANDIDATES


def test_report_slo_unmet_recommended_renders_warning_banner(
    sample_payload: SolvePayload,
) -> None:
    """Verify warning banner rendering when recommended layout does not meet SLO."""
    html_ok = render_html(sample_payload)
    assert "max_throughput" in html_ok
    assert "<th>Role</th>" in html_ok
    assert "<th>KV Cap / Prefix Hit Rate</th>" in html_ok

    payload: dict[str, object] = dict(sample_payload)
    assert sample_payload["recommended"] is not None
    recommended_ruled_out = dict(sample_payload["recommended"])
    recommended_ruled_out["slo_met"] = False
    recommended_ruled_out["slo_verdict"] = "ruled_out"
    payload["recommended"] = recommended_ruled_out
    payload["top_k"] = []
    payload["warnings"] = []
    html_unmet = render_html(payload)
    assert "INFEASIBLE" in html_unmet
    assert '<div class="alert-box alert-warning">' in html_unmet
    assert "recommended layout is infeasible" in html_unmet


def test_render_html_clean_svg_and_assumptions_card(
    sample_payload: SolvePayload,
) -> None:
    """Verify clean SVG formatting, dynamic chip specs, and safe disagg cmds."""
    payload = json.loads(json.dumps(sample_payload))
    if payload.get("recommended") and payload["recommended"].get("decode_pool"):
        payload["recommended"]["decode_pool"].pop("hbm_limit_gb", None)
    html = render_html(payload)
    assert "</text>'" not in html
    assert "94.76" in html
    assert "Assumptions, Hardware Specs" in html

    # Disaggregated commands must not concatenate `# ...; vllm serve` on one line.
    disagg_geom = from_hf_config(
        {
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
    )
    disagg_plan = solve_serving_plan(
        disagg_geom,
        Dtypes(kv=1.0),
        CHIPS["tpu7x"],
        TrafficScenario(
            name="disagg-check",
            isl=2048,
            osl=256,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(1,),
            kv_working_set_turns=1.0,
        ),
        SolveOptions(mode="disagg", max_chips=16),
    )
    disagg_payload = build_solve_payload(
        disagg_plan,
        chip=CHIPS["tpu7x"],
        dtypes=Dtypes(kv=1.0),
        model="test/m",
    )
    disagg_html = render_html(disagg_payload)
    assert "; # decode" not in disagg_html
    assert "; vllm serve" not in disagg_html


def test_report_card_renders_from_catalog(
    sample_payload: SolvePayload,
) -> None:
    """Verify report card and solve payload render from the chip catalog."""
    chip_spec = sample_payload["chip_spec"]
    assert "jax_version" in chip_spec
    assert isinstance(chip_spec["jax_version"], str)
    assert bool(chip_spec["jax_version"])
    assert "catalog" in chip_spec
    cat = chip_spec["catalog"]
    assert isinstance(cat, dict)
    assert "dcn_bw" in cat
    assert cat["dcn_bw"]["kind"] == "derived"
    assert "source" in cat["dcn_bw"]
    assert cat["hbm_bytes"]["display"] == "103.0 GB"
    assert cat["bf16_flops"]["display"] == "1,155 TFLOP/s"
    assert cat["fp8_flops"]["display"] == "2,300 TFLOP/s"
    assert "Peak BF16 matrix-multiply throughput" in cat["bf16_flops"]["description"]

    html = render_html(sample_payload)
    assert "12.50 GB/s" in html or "12.5 GB/s" in html
    assert "1,155 TFLOP/s" in html
    assert "2,300 TFLOP/s" in html
    assert "badge-fact" in html
    assert "fact" in html.lower()
    assert "derived" in html.lower()
    assert "assumed" in html.lower()

    custom_payload = json.loads(json.dumps(sample_payload))
    custom_payload["chip_spec"]["dcn_bw"] = 18.75e9
    custom_payload["chip_spec"]["pcie_bw"] = 32.0e9
    custom_payload["chip_spec"]["catalog"]["dcn_bw"]["value"] = 18.75e9
    custom_payload["chip_spec"]["catalog"]["pcie_bw"]["value"] = 32.0e9
    custom_html = render_html(custom_payload)
    assert "18.75 GB/s" in custom_html or "18.8 GB/s" in custom_html
    assert "32.0 GB/s" in custom_html


def test_executive_card_renders_bound_and_estimate_band(
    sample_payload: SolvePayload,
) -> None:
    """Verify executive summary card renders bound ~ est TTFT and TPOT bands."""
    html = render_html(sample_payload)
    exec_section = html.split('id="executive-summary"', 1)[1].split("</section>", 1)[0]
    assert "TTFT (bound ~ est)" in exec_section
    assert "TPOT (bound ~ est)" in exec_section
    assert " ~ " in exec_section

    # Assumptions card reflects chip_spec dynamically rather than hardcoding tpu7x.
    custom_chip = dc_replace(
        CHIPS["tpu7x"], name="custom_tpu_v6", hbm_bytes=32_000_000_000
    )
    custom_payload = dict(sample_payload)
    custom_payload["chip"] = "custom_tpu_v6"
    custom_payload["chip_spec"] = {
        "name": custom_chip.name,
        "hbm_bytes": custom_chip.hbm_bytes,
        "hbm_bw": custom_chip.hbm_bw,
        "bf16_flops": custom_chip.bf16_flops,
        "fp8_flops": custom_chip.fp8_flops,
        "vmem_bytes": custom_chip.vmem_bytes,
        "ici_link_bw": custom_chip.ici_link_bw,
        "ici_bw": custom_chip.ici_bw,
        "ici_hop_latency_s": custom_chip.ici_hop_latency_s,
    }
    custom_html = render_html(custom_payload)
    assert "custom_tpu_v6 Per-Core Specs:" in custom_html
    assert "32.0 GB" in custom_html


_BASE_POOL: dict[str, object] = {
    "instances": 1,
    "topology": "2x2x1",
    "tp": 1,
    "dp": 1,
    "ep": 1,
    "pp": 1,
    "pcp": 1,
    "kv_split": 1,
}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"tp": 2, "dp": 4, "ep": 8},
            "1x 2x2x1 (4 chips, 1 host, 8 cores): TP2 x DP4 x EP8",
        ),
        (
            {"pp": 8},
            "1x 2x2x1 (4 chips, 1 host, 8 cores): PP8",
        ),
        (
            {"pp": 2, "pcp": 4, "ep": 4},
            "1x 2x2x1 (4 chips, 1 host, 8 cores): EP4 x PP2 x PCP4",
        ),
        (
            {"instances": 4, "topology": "2x2x2", "tp": 16, "kv_split": 2, "ep": 16},
            "4x 2x2x2 (32 chips, 8 hosts, 64 cores): TP16 x EP16 x DCP2",
        ),
        ({"topology": "1x1x1"}, "1x 1x1x1: TP1"),
    ],
)
def test_pool_label_omits_single_count_and_lists_axes(
    overrides: dict[str, object],
    expected: str,
) -> None:
    """Pool label includes count prefix; only axes above 1 are listed."""
    assert pool_label({**_BASE_POOL, **overrides}) == expected


def test_render_html_caps_svg_background_circles(
    sample_payload: SolvePayload,
) -> None:
    """Verify SVG scatter caps background dots at <= 1200 and marks Top-k rings."""
    max_bg_dots = 1200
    payload: dict[str, object] = json.loads(json.dumps(sample_payload))
    cands = list(sample_payload.get("candidates", ()))
    base = dict(cands[0])
    spread: list[dict[str, object]] = []
    n_pts = 5000
    for i in range(n_pts):
        pt = dict(base)
        pt["concurrency"] = (i % 128) + 1
        pt["token_throughput_per_chip"] = 100.0 + float((i * 37) % 10000)
        pt["est_interactivity_tok_s"] = 10.0 + float((i * 53) % 1000)
        spread.append(pt)
    payload["candidates"] = spread
    html = render_html(payload)
    bg_circles = len(re.findall(r'<circle [^>]*fill="#cbd5e1"', html))
    assert 1 <= bg_circles <= max_bg_dots
    assert "#10b981" in html


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


def test_report_undecided_recommended_renders_funnel_counts(
    sample_payload: SolvePayload,
) -> None:
    """Render undecided recommendation without infeasible banner and show counts."""
    undecided_payload: dict[str, object] = dict(sample_payload)
    assert sample_payload["recommended"] is not None
    recommended_undecided = dict(sample_payload["recommended"])
    recommended_undecided["slo_met"] = False
    recommended_undecided["slo_verdict"] = "undecided"
    undecided_search = dict(sample_payload["search"])
    expected_not_ruled_out = 11
    expected_slo_met = 3
    undecided_search["not_ruled_out"] = expected_not_ruled_out
    undecided_search["slo_met"] = expected_slo_met
    undecided_scenario = dict(sample_payload["scenario"])
    undecided_scenario["tpot_slo_ms"] = 2.0
    expected_hbm_fit = 15
    undecided_search["hbm_fit"] = expected_hbm_fit
    undecided_payload["scenario"] = undecided_scenario
    undecided_payload["recommended"] = recommended_undecided
    undecided_payload["search"] = undecided_search
    undecided_payload["warnings"] = []
    html_undecided = render_html(undecided_payload)
    assert "INFEASIBLE" not in html_undecided
    assert "recommended layout is infeasible" not in html_undecided
    assert (
        '        <div class="stat-label">Not Ruled Out</div>\n'
        f'        <div class="stat-value">{expected_not_ruled_out}</div>\n'
        '        <div class="funnel-sub">'
        f"Ruled Out (bound &gt; SLO): {expected_hbm_fit - expected_not_ruled_out}<br>"
        f"Est. Meets SLO: {expected_slo_met} &middot; "
        f"Undecided: {expected_not_ruled_out - expected_slo_met}</div>"
    ) in html_undecided


def test_frontier_chart_plots_monotonic_2d_envelope_with_throughput_on_y(
    sample_payload: SolvePayload,
) -> None:
    """Frontier plots throughput on y and interactivity on x without 4D spikes."""
    payload = copy.deepcopy(sample_payload)
    # Include a lower-concurrency point at the same interactivity as a higher-tput point
    p_low = copy.deepcopy(payload["frontier"][0])
    p_low["concurrency"] = 1
    p_low["est_interactivity_tok_s"] = 1000.0
    p_low["token_throughput_per_chip"] = 2500.0
    p_high = copy.deepcopy(payload["frontier"][0])
    p_high["concurrency"] = 4
    p_high["est_interactivity_tok_s"] = 1000.0
    p_high["token_throughput_per_chip"] = 10000.0
    p_tput = copy.deepcopy(payload["frontier"][0])
    p_tput["concurrency"] = 64
    p_tput["est_interactivity_tok_s"] = 200.0
    p_tput["token_throughput_per_chip"] = 40000.0
    payload["frontier"] = [p_low, p_high, p_tput]
    payload["candidates"] = [p_low, p_high, p_tput]

    html = render_html(payload)
    chart = html[html.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert "Est. Interactivity (tok/s)</text>" in chart
    assert "Est. Token Throughput per Chip (tok/s/chip)</text>" in chart
    match = re.search(r'<polyline points="([^"]+)"', chart)
    assert match is not None
    coords = [
        tuple(float(v) for v in pair.split(",")) for pair in match.group(1).split()
    ]
    # Only the 2 true 2D envelope points remain; x rises as SVG y rises (lower tput)
    expected_envelope_len = 2
    assert len(coords) == expected_envelope_len
    assert coords[0][0] < coords[1][0]
    assert coords[0][1] < coords[1][1]


def test_top_layouts_table_uses_collapsed_launch_blocks_and_disagg_columns(
    sample_payload: SolvePayload,
) -> None:
    """Launch commands live in collapsed Copy blocks; disagg splits pool columns."""
    html = render_html(sample_payload)
    table = html[html.index('id="top-layouts-table"') :]
    table = table[: table.index("</section>")]
    assert "<th>Launch Command</th>" not in table
    assert "<th>Pool</th>" in table
    assert '<details class="launch"><summary>Launch command for #1' in table
    assert 'onclick="copyLaunch(this)"' in table

    # No-SLO funnel never overclaims "SLO Met"
    funnel = html[html.index('id="search-funnel"') :]
    funnel = funnel[: funnel.index("</section>")]
    if (
        sample_payload["scenario"].get("ttft_slo_ms") is None
        and sample_payload["scenario"].get("tpot_slo_ms") is None
    ):
        assert "No SLO set" in funnel
        assert "SLO Met" not in funnel


def test_report_renders_five_stage_funnel_and_submodule_pools(
    sample_payload: SolvePayload,
) -> None:
    """HTML report renders 5-stage funnel, mixed SLO ring colors, and Attn | MoE."""
    moe_payload = copy.deepcopy(sample_payload)
    moe_payload["has_moe"] = True
    moe_search = dict(moe_payload["search"])
    moe_search["core_grid"] = 2940
    moe_search["enumerated"] = 98
    moe_search["hbm_fit"] = 98
    moe_search["not_ruled_out"] = 98
    moe_payload["search"] = moe_search
    moe_payload["budget"] = 4
    moe_payload["chip_budget"] = {"max_chips": 4, "budget_source": "flag"}
    # Include an undecided frontier point higher than the estimate_meets Rank #1 point
    rank1_point = copy.deepcopy(moe_payload["top_k"][0]["point"])
    rank1_point["slo_verdict"] = "estimate_meets"
    rank1_point["concurrency"] = 64
    undecided_point = copy.deepcopy(rank1_point)
    undecided_point["slo_verdict"] = "undecided"
    undecided_point["token_throughput_per_chip"] = (
        float(rank1_point["token_throughput_per_chip"]) * 1.15
    )
    moe_payload["top_k"][0]["point"] = rank1_point
    moe_payload["top_k"][0]["frontier_role"] = "max_throughput"
    moe_payload["frontier"] = [rank1_point, undecided_point]
    moe_payload["candidates"] = [rank1_point, undecided_point]

    html = render_html(moe_payload)
    funnel = html[html.index('id="search-funnel"') :]
    funnel = funnel[: funnel.index("</section>")]
    assert "Core-Valid Grid" in funnel
    assert "2,940" in funnel
    assert "Architecture &amp; Runtime Legal" in funnel
    assert "Attn:" in html
    assert "MoE:" in html
    assert "1 hosts" not in html
    chart = html[html.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert "#1 (c=64)" in chart
    assert "#1 (min bound" not in chart


def test_report_honest_l1_headings_funnel_and_excludes_ruled_out_frontier(
    sample_payload: SolvePayload,
) -> None:
    """Verify honest L1 headings, HBM-stage drop wording, and ruled-out exclusion."""
    html_base = render_html(sample_payload)
    assert "Top-Ranked L1 Candidate (Uncalibrated Shortlist)" in html_base
    assert "Top-Ranked L1 Candidate Launch Command (Uncalibrated)" in html_base
    assert "Uncalibrated L1 per-resource decode step lower bounds" in html_base
    assert "Executive Recommendation" not in html_base
    assert "Recommended Launch Command" not in html_base
    assert "Best Parallelism:" not in html_base
    assert "SMEM page-table" in html_base
    assert "prefill logit workspace" in html_base
    assert 'id="bound-families-table"' in html_base

    payload_no_families = copy.deepcopy(sample_payload)
    payload_no_families.pop("bound_families_by_concurrency", None)
    html_no_families = render_html(payload_no_families)
    assert 'id="bound-families-table"' not in html_no_families

    payload = copy.deepcopy(sample_payload)
    ruled_out_pts = []
    for pt in payload["frontier"]:
        copy_pt = dict(pt)
        copy_pt["slo_verdict"] = "ruled_out"
        copy_pt["slo_met"] = False
        ruled_out_pts.append(copy_pt)
    payload["frontier"] = ruled_out_pts
    html_ruled = render_html(payload)
    funnel = html_ruled[html_ruled.index('id="search-funnel"') :]
    funnel = funnel[: funnel.index("</section>")]
    assert (
        '<div class="stat-label" style="color:var(--primary);">'
        'Est. 2D Frontier</div><div class="stat-value" '
        'style="color:var(--primary);">0</div>'
    ) in funnel.replace("\n        ", "")
    chart = html_ruled[html_ruled.index('id="frontier-chart"') :]
    chart = chart[: chart.index("</section>")]
    assert "<polyline" not in chart


def test_report_renders_optional_reserve_provenance_assumptions_and_families(
    sample_payload: SolvePayload,
) -> None:
    """Verify guarded rendering when HBM reserve, provenance, and families exist."""
    payload = copy.deepcopy(sample_payload)
    reserve_gb = 12.0
    expected_usable = 103.0 * 0.92 - reserve_gb
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
        "basis": "L1 analytical bounds and uncalibrated serial estimates",
        "ranking_basis": "bound_ordered_handoff",
        "supplied_values": {
            "hbm_reserve_gb": {
                "flag": "--hbm-reserve-gb",
                "value": reserve_gb,
                "kind": "assumed",
            },
        },
    }
    payload["solver_assumptions"] = [
        {"name": "moe_activation", "value": "dense_fallback", "kind": "derived"},
    ]
    sample_point = copy.deepcopy(payload["frontier"][0])
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
    assert "Top-Ranked Single-Host Candidate:" in html
    assert "--hbm-reserve-gb=12.0" in html
    assert "bound_ordered_handoff" in html
    assert "Solver Assumptions:" in html
    assert "moe_activation=dense_fallback" in html
    assert "Bound-Ordered Candidate Families (L1 Handoff)" in html
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
        "hbm_limit_gb": 94.76,
        "best_parallelism": None,
    }
    html = render_html(payload)
    assert "Top-Ranked Single-Host Candidate: <code>N/A</code>" in html
    assert "<code>None</code>" not in html


def test_report_renders_live_solve_handoff_families(
    sample_payload: SolvePayload,
) -> None:
    """Verify real build_solve_payload output populates bound_families HTML section."""
    assert "bound_families_by_concurrency" in sample_payload
    assert "solver_assumptions" in sample_payload
    html = render_html(sample_payload)
    assert "Bound-Ordered Candidate Families (L1 Handoff)" in html
    assert 'id="bound-families-table"' in html
    assert "Solver Assumptions:" in html
