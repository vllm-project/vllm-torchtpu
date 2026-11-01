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
"""Standalone HTML report renderer for analytical roofline solve/v1 payloads."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

MAX_EMBEDDED_CANDIDATES: int = 2000


def _escape(val: Any) -> str:
    """Safe HTML entity escape for string or numeric values."""
    if val is None:
        return ""
    return html.escape(str(val))


def _render_frontier_svg(
    points: list[dict[str, Any]], frontier: list[dict[str, Any]]
) -> str:
    """Inline SVG Pareto frontier scatter plot (Throughput vs Interactivity)."""
    width = 720
    height = 360
    pad_left = 70
    pad_right = 30
    pad_top = 30
    pad_bottom = 50

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_pts = points if points else frontier
    if not all_pts:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg">'
            f'<text x="{width // 2}" y="{height // 2}" '
            'text-anchor="middle" fill="#64748b">'
            "No candidate points available</text></svg>"
        )

    max_x = max(p.get("token_throughput_per_chip", 0.0) for p in all_pts) or 1.0
    max_y = max(p.get("p90_interactivity_tok_s", 0.0) for p in all_pts) or 1.0
    max_x *= 1.15
    max_y *= 1.15

    def scale_x(x: float) -> float:
        return pad_left + (x / max_x) * plot_w

    def scale_y(y: float) -> float:
        return pad_top + plot_h - (y / max_y) * plot_h

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" '
        'fill="#f8fafc" stroke="#e2e8f0"/>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" '
        f'x2="{pad_left + plot_w}" y2="{pad_top + plot_h}" '
        'stroke="#94a3b8" stroke-width="1.5"/>',
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{pad_top + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>',
        f'<text x="{pad_left + plot_w // 2}" y="{height - 12}" '
        'text-anchor="middle" font-size="12" fill="#475569">'
        "Token Throughput per Chip (tok/s/chip)</text>",
        f'<text x="18" y="{pad_top + plot_h // 2}" text-anchor="middle" '
        'font-size="12" fill="#475569" '
        f'transform="rotate(-90, 18, {pad_top + plot_h // 2})">'
        "p90 Interactivity (tok/s)</text>",
    ]

    for i in range(5):
        vx = (max_x / 4) * i
        vy = (max_y / 4) * i
        px = scale_x(vx)
        py = scale_y(vy)
        svg_parts.append(
            f'<line x1="{px}" y1="{pad_top}" x2="{px}" '
            f'y2="{pad_top + plot_h}" stroke="#e2e8f0" stroke-dasharray="3,3"/>'
        )
        svg_parts.append(
            f'<text x="{px}" y="{pad_top + plot_h + 18}" text-anchor="middle" '
            f'font-size="10" fill="#64748b">{vx:.0f}</text>'
        )
        svg_parts.append(
            f'<line x1="{pad_left}" y1="{py}" x2="{pad_left + plot_w}" '
            f'y2="{py}" stroke="#e2e8f0" stroke-dasharray="3,3"/>'
        )
        svg_parts.append(
            f'<text x="{pad_left - 8}" y="{py + 4}" text-anchor="end" '
            f'font-size="10" fill="#64748b">{vy:.0f}</text>'
        )

    for p in all_pts:
        cx = scale_x(p.get("token_throughput_per_chip", 0.0))
        cy = scale_y(p.get("p90_interactivity_tok_s", 0.0))
        t_tip = (
            f"c={p.get('concurrency', 1)} | "
            f"{p.get('token_throughput_per_chip', 0.0):.1f} tok/s/chip | "
            f"{p.get('p90_interactivity_tok_s', 0.0):.1f} tok/s"
        )
        svg_parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="#cbd5e1" '
            'stroke="#94a3b8" stroke-width="0.5" opacity="0.7">'
            f"<title>{_escape(t_tip)}</title></circle>"
        )

    sorted_f = sorted(frontier, key=lambda p: p.get("token_throughput_per_chip", 0.0))
    if len(sorted_f) >= 2:
        poly_pts = " ".join(
            f"{scale_x(p.get('token_throughput_per_chip', 0.0)):.1f},"
            f"{scale_y(p.get('p90_interactivity_tok_s', 0.0)):.1f}"
            for p in sorted_f
        )
        svg_parts.append(
            f'<polyline points="{poly_pts}" fill="none" '
            'stroke="#2563eb" stroke-width="2.5"/>'
        )

    for p in sorted_f:
        cx = scale_x(p.get("token_throughput_per_chip", 0.0))
        cy = scale_y(p.get("p90_interactivity_tok_s", 0.0))
        f_tip = (
            f"Frontier: c={p.get('concurrency', 1)} | "
            f"{p.get('token_throughput_per_chip', 0.0):.1f} tok/s/chip | "
            f"{p.get('p90_interactivity_tok_s', 0.0):.1f} tok/s | "
            f"TTFT: {p.get('p90_ttft_ms', 0.0):.1f}ms | "
            f"TPOT: {p.get('p90_tpot_ms', 0.0):.2f}ms"
        )
        svg_parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="#1d4ed8" '
            'stroke="#ffffff" stroke-width="1.5">'
            f"<title>{_escape(f_tip)}</title></circle>"
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _render_limits_svg(floor: dict[str, Any] | None) -> str:
    """Inline horizontal bar chart for decode-step resource times (ms)."""
    width = 720
    height = 240
    pad_left = 130
    pad_right = 90
    pad_top = 25

    if not floor:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg">'
            f'<text x="{width // 2}" y="{height // 2}" '
            'text-anchor="middle" fill="#64748b">'
            "No decode floor data available</text></svg>"
        )

    items = [
        ("HBM", floor.get("hbm_ms", 0.0), "#3b82f6"),
        ("MatMul", floor.get("matmul_ms", 0.0), "#10b981"),
        ("ICI Comm", floor.get("ici_ms", 0.0), "#f59e0b"),
        ("Op Chain", floor.get("chain_ms", 0.0), "#8b5cf6"),
        ("Hop Latency", floor.get("hop_latency_ms", 0.0), "#ec4899"),
        ("Hard Floor", floor.get("hard_floor_ms", 0.0), "#ef4444"),
    ]

    max_val = max((val for _, val, _ in items), default=1.0) or 1.0
    max_val *= 1.25
    plot_w = width - pad_left - pad_right
    bar_h = 22
    bar_gap = 12

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]

    for i, (label, val, color) in enumerate(items):
        y = pad_top + i * (bar_h + bar_gap)
        bar_w = max(2.0, (val / max_val) * plot_w)
        is_hard = label == "Hard Floor"
        weight = "bold" if is_hard else "normal"
        svg_parts.append(
            f'<text x="{pad_left - 12}" y="{y + 15}" text-anchor="end" '
            f'font-size="12" font-weight="{weight}" fill="#334155">'
            f"{_escape(label)}</text>'"
        )
        svg_parts.append(
            f'<rect x="{pad_left}" y="{y}" width="{bar_w:.1f}" '
            f'height="{bar_h}" fill="{color}" rx="3">'
            f"<title>{_escape(label)}: {val:.3f} ms</title></rect>"
        )
        svg_parts.append(
            f'<text x="{pad_left + bar_w + 8}" y="{y + 15}" font-size="11" '
            f'font-weight="{weight}" fill="#475569">{val:.3f} ms</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _render_memory_svg(pools: list[dict[str, Any]]) -> str:
    """Inline stacked horizontal bar chart for HBM breakdown vs capacity."""
    width = 720
    row_h = 60
    pad_left = 130
    pad_right = 110
    pad_top = 25
    height = pad_top + max(1, len(pools)) * row_h + 35
    plot_w = width - pad_left - pad_right

    if not pools:
        return (
            f'<svg viewBox="0 0 {width} 120" class="chart-svg">'
            f'<text x="{width // 2}" y="60" text-anchor="middle" fill="#64748b">'
            "No pool memory data available</text></svg>"
        )

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]

    for i, pool in enumerate(pools):
        y = pad_top + i * row_h
        role = pool.get("role", "pool")
        topo = pool.get("topology", "")
        limit_gb = pool.get("hbm_limit_gb", 32.0) or 32.0
        w_gb = pool.get("hbm_weights_gb", 0.0)
        kv_gb = pool.get("hbm_kv_gb", 0.0)
        st_gb = pool.get("hbm_state_gb", 0.0)
        used_gb = pool.get("hbm_used_gb", w_gb + kv_gb + st_gb)
        free_gb = max(0.0, limit_gb - used_gb)

        label = f"{role.capitalize()} ({topo})"
        svg_parts.append(
            f'<text x="{pad_left - 12}" y="{y + 20}" text-anchor="end" '
            f'font-size="12" fill="#334155">{_escape(label)}</text>'
        )

        w_w = (w_gb / limit_gb) * plot_w
        kv_w = (kv_gb / limit_gb) * plot_w
        st_w = (st_gb / limit_gb) * plot_w
        free_w = max(0.0, plot_w - (w_w + kv_w + st_w))

        bar_h = 24
        cur_x = float(pad_left)
        if w_w > 0:
            svg_parts.append(
                f'<rect x="{cur_x:.1f}" y="{y}" width="{w_w:.1f}" '
                f'height="{bar_h}" fill="#3b82f6">'
                f"<title>Weights: {w_gb:.2f} GB</title></rect>"
            )
            cur_x += w_w
        if kv_w > 0:
            svg_parts.append(
                f'<rect x="{cur_x:.1f}" y="{y}" width="{kv_w:.1f}" '
                f'height="{bar_h}" fill="#10b981">'
                f"<title>KV Cache: {kv_gb:.2f} GB</title></rect>"
            )
            cur_x += kv_w
        if st_w > 0:
            svg_parts.append(
                f'<rect x="{cur_x:.1f}" y="{y}" width="{st_w:.1f}" '
                f'height="{bar_h}" fill="#f59e0b">'
                f"<title>Recurrent State: {st_gb:.2f} GB</title></rect>"
            )
            cur_x += st_w
        if free_w > 0:
            svg_parts.append(
                f'<rect x="{cur_x:.1f}" y="{y}" width="{free_w:.1f}" '
                f'height="{bar_h}" fill="#e2e8f0">'
                f"<title>Headroom: {free_gb:.2f} GB</title></rect>"
            )

        svg_parts.append(
            f'<rect x="{pad_left}" y="{y}" width="{plot_w}" height="{bar_h}" '
            'fill="none" stroke="#94a3b8" rx="3"/>'
        )
        util_pct = (used_gb / limit_gb) * 100.0 if limit_gb > 0 else 0.0
        svg_parts.append(
            f'<text x="{pad_left + plot_w + 10}" y="{y + 17}" font-size="11" '
            f'fill="#475569">{used_gb:.1f} / {limit_gb:.1f} GB ({util_pct:.0f}%)</text>'
        )

    legend_y = height - 16
    svg_parts.append(
        f'<g transform="translate({pad_left}, {legend_y})">'
        '<rect x="0" y="-8" width="10" height="10" fill="#3b82f6"/>'
        '<text x="14" y="0" font-size="11" fill="#475569">Weights</text>'
        '<rect x="80" y="-8" width="10" height="10" fill="#10b981"/>'
        '<text x="94" y="0" font-size="11" fill="#475569">KV Cache</text>'
        '<rect x="170" y="-8" width="10" height="10" fill="#f59e0b"/>'
        '<text x="184" y="0" font-size="11" fill="#475569">State</text>'
        '<rect x="240" y="-8" width="10" height="10" fill="#e2e8f0" '
        'stroke="#94a3b8"/>'
        '<text x="254" y="0" font-size="11" fill="#475569">Free Headroom</text>'
        "</g>"
    )
    svg_parts.append("</svg>")
    return "".join(svg_parts)


CSS_CONTENT = """
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-card: #ffffff;
  --border: #e2e8f0;
  --text-main: #0f172a;
  --text-muted: #64748b;
  --primary: #2563eb;
  --primary-light: #eff6ff;
  --success: #10b981;
  --warning-bg: #fffbeb;
  --warning-border: #fef3c7;
  --warning-text: #92400e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif;
  background-color: var(--bg-secondary);
  color: var(--text-main);
  line-height: 1.5;
  padding: 24px;
}
.container {
  max-width: 1100px;
  margin: 0 auto;
}
header {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
h1 { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
h2 { font-size: 18px; font-weight: 600; margin-bottom: 14px; color: #1e293b; }
.meta { font-size: 13px; color: var(--text-muted); }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  background-color: var(--primary-light);
  color: var(--primary);
  margin-right: 6px;
}
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.grid-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  margin-top: 14px;
}
.stat-box {
  background: var(--bg-secondary);
  padding: 12px 14px;
  border-radius: 6px;
  border: 1px solid var(--border);
}
.stat-label {
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  font-weight: 600;
}
.stat-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-main);
  margin-top: 2px;
}
.alert-box {
  padding: 12px 16px;
  border-radius: 6px;
  margin-bottom: 20px;
  font-size: 13px;
}
.alert-warning {
  background: var(--warning-bg);
  border: 1px solid var(--warning-border);
  color: var(--warning-text);
}
.funnel-container {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}
.funnel-step {
  flex: 1;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 14px;
  text-align: center;
}
.funnel-arrow { color: var(--text-muted); font-size: 16px; }
.chart-container {
  display: flex;
  justify-content: center;
  margin: 10px 0;
}
.chart-svg {
  width: 100%;
  max-width: 720px;
  height: auto;
  font-family: inherit;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-top: 10px;
}
th, td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
th {
  background: var(--bg-secondary);
  font-weight: 600;
  color: var(--text-muted);
}
tr:hover td { background: #f1f5f9; }
.code-box {
  background: #0f172a;
  color: #f8fafc;
  border-radius: 6px;
  padding: 14px;
  margin-top: 16px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  overflow-x: auto;
}
.code-header {
  font-size: 11px;
  color: #94a3b8;
  margin-bottom: 6px;
  text-transform: uppercase;
  font-weight: 600;
}
.cmd-cell {
  display: inline-block;
  max-width: 250px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}
footer {
  margin-top: 30px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-muted);
  text-align: center;
}
"""


def render_html(payload: dict[str, Any]) -> str:
    """Render a fully self-contained, offline HTML report from a solve/v1 payload."""
    rec = payload.get("recommended")
    frontier = payload.get("frontier", ())
    candidates = payload.get("candidates", ())
    search = payload.get("search", {})
    scenario = payload.get("scenario", {})
    warnings = payload.get("warnings", ())
    top_k = payload.get("top_k", [])
    model = payload.get("model") or "Model"
    chip = payload.get("chip") or "TPU"

    embedded_payload = dict(payload)
    raw_cands = embedded_payload.get("candidates", ())
    if len(raw_cands) > MAX_EMBEDDED_CANDIDATES:
        stride = math.ceil(len(raw_cands) / MAX_EMBEDDED_CANDIDATES)
        embedded_payload["candidates"] = raw_cands[::stride][:MAX_EMBEDDED_CANDIDATES]

    embedded_json = json.dumps(embedded_payload, indent=2).replace("</", "<\\/")

    if rec is not None and not rec.get("slo_met", True) and not warnings:
        warnings = (
            "No candidate layout met latency SLOs; recommended layout is infeasible.",
        )

    if rec is not None and rec.get("slo_met", True):
        rec_mode = str(rec.get("mode", "")).upper()
        rec_chips = rec.get("total_chips", 0)
        rec_c = rec.get("concurrency", 1)
        rec_tput = f"{rec.get('token_throughput_per_chip', 0.0):.1f}"
        rec_int = f"{rec.get('p90_interactivity_tok_s', 0.0):.1f}"
        rec_ttft = f"{rec.get('p90_ttft_ms', 0.0):.1f}"
        rec_tpot = f"{rec.get('p90_tpot_ms', 0.0):.2f}"
        rec_binding = str(rec.get("binding", "hbm"))
        decode_floor = rec.get("decode_floor")

        pools_list: list[dict[str, Any]] = []
        if rec.get("prefill_pool"):
            pools_list.append(rec["prefill_pool"])
        if rec.get("decode_pool"):
            pools_list.append(rec["decode_pool"])
        pools_summary = " + ".join(
            f"{p.get('instances', 1)}x {p.get('topology', '')} ({p.get('role', '')})"
            for p in pools_list
        )
    else:
        rec_mode = "INFEASIBLE"
        rec_chips = 0
        rec_c = 0
        rec_tput = "0.0"
        rec_int = "0.0"
        rec_ttft = "-"
        rec_tpot = "-"
        rec_binding = "None"
        decode_floor = None
        pools_list = []
        pools_summary = "No feasible layout met constraints"

    frontier_svg = _render_frontier_svg(list(candidates), list(frontier))
    limits_svg = _render_limits_svg(decode_floor)
    memory_svg = _render_memory_svg(pools_list)

    enum_cnt = search.get("enumerated", 0) if search else 0
    hbm_cnt = search.get("hbm_fit", 0) if search else 0
    slo_cnt = search.get("slo_met", 0) if search else 0
    front_cnt = search.get("frontier", 0) if search else 0

    warnings_html = ""
    if warnings:
        warn_items = "".join(f"<li>{_escape(w)}</li>" for w in warnings)
        warnings_html = (
            f'<div class="alert-box alert-warning">'
            f"<strong>Warnings:</strong><ul>{warn_items}</ul></div>"
        )

    launch_cmds_html = ""
    if top_k and top_k[0].get("launch", {}).get("commands"):
        cmds = top_k[0]["launch"]["commands"]
        cmds_str = "\n".join(cmds)
        launch_cmds_html = (
            '<div class="code-box">'
            '<div class="code-header">Recommended Launch Command</div>'
            f"<pre><code>{_escape(cmds_str)}</code></pre></div>"
        )

    table_rows = []
    for rank, entry in enumerate(top_k, 1):
        pt = entry.get("point", {})
        launch = entry.get("launch", {})
        role_val = str(entry.get("frontier_role", "candidate"))
        margin_val = float(entry.get("margin_vs_rank1_pct", 0.0))
        mode_val = pt.get("mode", "")
        c_val = pt.get("concurrency", 1)
        chips_val = pt.get("total_chips", 0)
        int_val = f"{pt.get('p90_interactivity_tok_s', 0.0):.1f}"
        tput_val = f"{pt.get('token_throughput_per_chip', 0.0):.1f}"
        ttft_val = f"{pt.get('p90_ttft_ms', 0.0):.1f}"
        tpot_val = f"{pt.get('p90_tpot_ms', 0.0):.2f}"
        cap_raw = pt.get("kv_cache_capacity_tokens")
        cap_str = f"{int(cap_raw):,}" if cap_raw is not None else "inf"
        eff_hit = float(pt.get("effective_prefix_hit_rate", 0.0))
        cmd_str = "; ".join(launch.get("commands", []))

        p_pools = []
        if pt.get("prefill_pool"):
            p_pools.append(
                f"{pt['prefill_pool'].get('instances', 1)}x"
                f"{pt['prefill_pool'].get('topology', '')}"
            )
        if pt.get("decode_pool"):
            p_pools.append(
                f"{pt['decode_pool'].get('instances', 1)}x"
                f"{pt['decode_pool'].get('topology', '')}"
            )
        p_str = "+".join(p_pools)

        table_rows.append(
            f"<tr>"
            f"<td>#{rank}</td>"
            f"<td><code>{_escape(role_val)}</code> ({margin_val:+.1f}%)</td>"
            f"<td><strong>{_escape(mode_val)}</strong></td>"
            f"<td>{c_val}</td>"
            f"<td>{chips_val}</td>"
            f"<td><code>{_escape(p_str)}</code></td>"
            f"<td>{int_val}</td>"
            f"<td>{tput_val}</td>"
            f"<td>{ttft_val}</td>"
            f"<td>{tpot_val}</td>"
            f"<td>{_escape(cap_str)} / {eff_hit:.2f}</td>"
            f'<td><code class="cmd-cell" title="{_escape(cmd_str)}">'
            f"{_escape(cmd_str)}</code></td>"
            f"</tr>"
        )
    top_table_body = (
        "".join(table_rows)
        if table_rows
        else (
            '<tr><td colspan="12" style="text-align:center;">'
            "No feasible layouts found</td></tr>"
        )
    )

    scen_name = _escape(scenario.get("name", "custom"))
    scen_isl = scenario.get("isl", 0)
    scen_osl = scenario.get("osl", 0)
    scen_hit = scenario.get("prefix_hit_rate", 0.0)

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>',
        f"<title>Roofline Report - {_escape(model)} on {_escape(chip)}</title>",
        f"<style>{CSS_CONTENT}</style>",
        "</head>",
        "<body>",
        '<div class="container">',
        "  <header>",
        "    <h1>TPU Roofline Solve Report</h1>",
        '    <div class="meta">',
        f'      <span class="badge">{_escape(chip)}</span>',
        f"      <strong>Model:</strong> {_escape(model)} &bull; ",
        f"      <strong>Scenario:</strong> {scen_name} "
        f"(ISL={scen_isl}, OSL={scen_osl}, hit_rate={scen_hit:.2f})",
        "    </div>",
        "  </header>",
        f"  {warnings_html}",
        '  <section id="executive-summary" class="card">',
        "    <h2>Executive Recommendation</h2>",
        "    <div>",
        '      <span class="badge" style="background:#dbeafe; color:#1e40af; '
        f'font-size:13px; padding:4px 10px;">{_escape(rec_mode)}</span>',
        f"      <strong>{_escape(pools_summary)}</strong> &bull; "
        f"Concurrency {rec_c} across {rec_chips} chips",
        "    </div>",
        '    <div class="grid-stats">',
        '      <div class="stat-box">',
        '        <div class="stat-label">Throughput / Chip</div>',
        f'        <div class="stat-value">{rec_tput} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "tok/s/chip</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">p90 Interactivity</div>',
        f'        <div class="stat-value">{rec_int} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "tok/s</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">p90 TTFT</div>',
        f'        <div class="stat-value">{rec_ttft} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "ms</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">p90 TPOT</div>',
        f'        <div class="stat-value">{rec_tpot} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "ms</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">Binding Unit</div>',
        '        <div class="stat-value" style="font-size:16px;">'
        f"{_escape(rec_binding)}</div>",
        "      </div>",
        "    </div>",
        f"    {launch_cmds_html}",
        "  </section>",
        '  <section id="search-funnel" class="card">',
        "    <h2>Search Funnel</h2>",
        '    <div class="funnel-container">',
        '      <div class="funnel-step">',
        '        <div class="stat-label">Enumerated</div>',
        f'        <div class="stat-value">{enum_cnt}</div>',
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        '      <div class="funnel-step">',
        '        <div class="stat-label">HBM Fit</div>',
        f'        <div class="stat-value">{hbm_cnt}</div>',
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        '      <div class="funnel-step">',
        '        <div class="stat-label">SLO Met</div>',
        f'        <div class="stat-value">{slo_cnt}</div>',
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        '      <div class="funnel-step" style="border-color:var(--primary); '
        'background:var(--primary-light);">',
        '        <div class="stat-label" style="color:var(--primary);">'
        "Pareto Frontier</div>",
        '        <div class="stat-value" style="color:var(--primary);">'
        f"{front_cnt}</div>",
        "      </div>",
        "    </div>",
        "  </section>",
        '  <section id="frontier-chart" class="card">',
        "    <h2>Pareto Frontier (Interactivity vs. Throughput)</h2>",
        f'    <div class="chart-container">{frontier_svg}</div>',
        "  </section>",
        '  <section id="limits-chart" class="card">',
        "    <h2>Decode Step Resource Times</h2>",
        f'    <div class="chart-container">{limits_svg}</div>',
        "  </section>",
        '  <section id="memory-chart" class="card">',
        "    <h2>HBM Memory Footprint vs. Capacity</h2>",
        f'    <div class="chart-container">{memory_svg}</div>',
        "  </section>",
        '  <section id="top-layouts-table" class="card">',
        "    <h2>Top Layout Options</h2>",
        "    <table>",
        "      <thead>",
        "        <tr>",
        "          <th>Rank</th>",
        "          <th>Role</th>",
        "          <th>Mode</th>",
        "          <th>Conc</th>",
        "          <th>Chips</th>",
        "          <th>Pools</th>",
        "          <th>Int (tok/s)</th>",
        "          <th>Tput (tok/s/chip)</th>",
        "          <th>TTFT (ms)</th>",
        "          <th>TPOT (ms)</th>",
        "          <th>KV Cap / Hit</th>",
        "          <th>Launch Command</th>",
        "        </tr>",
        "      </thead>",
        "      <tbody>",
        f"        {top_table_body}",
        "      </tbody>",
        "    </table>",
        "  </section>",
        "  <footer>",
        "    Analytical roofline report generated by "
        "<code>vllm_torchtpu.roofline</code> &bull; "
        "All metrics are modeled analytical floors",
        "  </footer>",
        "</div>",
        '<script type="application/json" id="roofline-data">',
        f"{embedded_json}",
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(html_parts)


def write_html(payload: dict[str, Any], path: Path | str) -> Path:
    """Render and write HTML report to disk at path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_html(payload)
    dest.write_text(rendered, encoding="utf-8")
    return dest
