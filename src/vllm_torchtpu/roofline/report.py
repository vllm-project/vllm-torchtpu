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
"""Standalone HTML report renderer for analytical roofline solve/v2 payloads."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .chips import CHIPS, ChipSpec, chip_catalog, format_constant_display
from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    MAX_EMBEDDED_CANDIDATES,
    MAX_SVG_BG_POINTS,
    MIN_FRONTIER_POINTS_FOR_POLYLINE,
    PREFIX_RANGE_PAIR_LEN,
    SVG_GRID_CELL_PX,
    UNIT_GIGA,
    UNIT_MEGA,
    UNIT_PETA,
    UNIT_TERA,
)
from .emit import pool_label


def _as_mapping(value: object) -> Mapping[str, object]:
    """Return value as a mapping when compatible with JSON objects, else empty."""
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> list[object]:
    """Return value as a list for non-string sequences, else empty list."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce numeric JSON scalar to float safely."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def _as_int(value: object, default: int = 0) -> int:
    """Coerce numeric JSON scalar to int safely."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def _escape(val: object) -> str:
    """Escape untrusted text for safe inline HTML embedding."""
    if val is None:
        return ""
    return html.escape(str(val))


def _downsample_svg_points(
    points: Sequence[Mapping[str, object]],
    bounds: tuple[float, float],
    plot_size: tuple[int, int],
) -> list[Mapping[str, object]]:
    """Deduplicate dense scatter points on a 2D pixel grid while preserving envelope."""
    max_x, max_y = bounds
    plot_w, plot_h = plot_size
    if len(points) <= MAX_SVG_BG_POINTS:
        return list(points)
    seen: set[tuple[int, int]] = set()
    deduped: list[Mapping[str, object]] = []
    for p in points:
        px = (_as_float(p.get("token_throughput_per_chip")) / max_x) * plot_w
        py = (_as_float(p.get("est_interactivity_tok_s")) / max_y) * plot_h
        cell = (int(px // SVG_GRID_CELL_PX), int(py // SVG_GRID_CELL_PX))
        if cell not in seen:
            seen.add(cell)
            deduped.append(p)
    if len(deduped) <= MAX_SVG_BG_POINTS:
        return deduped
    stride = math.ceil(len(deduped) / MAX_SVG_BG_POINTS)
    return deduped[::stride][:MAX_SVG_BG_POINTS]


def _render_frontier_svg(
    points: Sequence[Mapping[str, object]],
    frontier: Sequence[Mapping[str, object]],
    top_k: Sequence[Mapping[str, object]] = (),
) -> str:
    """Render inline SVG Pareto frontier scatter plot."""
    width, height = 720, 360
    pad_left, pad_right, pad_top, pad_bottom = 70, 30, 30, 50
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

    max_x = (
        max(
            (_as_float(p.get("token_throughput_per_chip")) for p in all_pts),
            default=0.0,
        )
        or 1.0
    ) * 1.15
    max_y = (
        max(
            (_as_float(p.get("est_interactivity_tok_s")) for p in all_pts),
            default=0.0,
        )
        or 1.0
    ) * 1.15

    def scale_x(x: float) -> float:
        """Scale X data coordinate to SVG coordinate."""
        return pad_left + (x / max_x) * plot_w

    def scale_y(y: float) -> float:
        """Scale Y data coordinate to SVG coordinate."""
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
        "Est. Interactivity (tok/s)</text>",
    ]

    for i in range(5):
        vx = (max_x / 4) * i
        vy = (max_y / 4) * i
        px, py = scale_x(vx), scale_y(vy)
        svg_parts.extend(
            [
                f'<line x1="{px}" y1="{pad_top}" x2="{px}" '
                f'y2="{pad_top + plot_h}" stroke="#e2e8f0" stroke-dasharray="3,3"/>',
                f'<text x="{px}" y="{pad_top + plot_h + 18}" text-anchor="middle" '
                f'font-size="10" fill="#64748b">{vx:.0f}</text>',
                f'<line x1="{pad_left}" y1="{py}" x2="{pad_left + plot_w}" '
                f'y2="{py}" stroke="#e2e8f0" stroke-dasharray="3,3"/>',
                f'<text x="{pad_left - 8}" y="{py + 4}" text-anchor="end" '
                f'font-size="10" fill="#64748b">{vy:.0f}</text>',
            ]
        )

    for p in _downsample_svg_points(all_pts, (max_x, max_y), (plot_w, plot_h)):
        tput = _as_float(p.get("token_throughput_per_chip"))
        inter = _as_float(p.get("est_interactivity_tok_s"))
        svg_parts.append(
            f'<circle cx="{scale_x(tput):.1f}" cy="{scale_y(inter):.1f}" r="3" '
            'fill="#cbd5e1" stroke="#94a3b8" stroke-width="0.5" opacity="0.7"/>'
        )

    sorted_f = sorted(
        frontier, key=lambda p: _as_float(p.get("token_throughput_per_chip"))
    )
    if len(sorted_f) >= MIN_FRONTIER_POINTS_FOR_POLYLINE:
        poly_pts = " ".join(
            f"{scale_x(_as_float(p.get('token_throughput_per_chip'))):.1f},"
            f"{scale_y(_as_float(p.get('est_interactivity_tok_s'))):.1f}"
            for p in sorted_f
        )
        svg_parts.append(
            f'<polyline points="{poly_pts}" fill="none" '
            'stroke="#2563eb" stroke-width="2.5"/>'
        )

    for p in sorted_f:
        tput = _as_float(p.get("token_throughput_per_chip"))
        inter = _as_float(p.get("est_interactivity_tok_s"))
        f_tip = (
            f"Frontier: concurrency={_as_int(p.get('concurrency'), 1)} | "
            f"{tput:.1f} tok/s/chip | {inter:.1f} tok/s | "
            f"TTFT: {_as_float(p.get('est_ttft_ms')):.1f}ms | "
            f"TPOT: {_as_float(p.get('est_tpot_ms')):.2f}ms"
        )
        svg_parts.append(
            f'<circle cx="{scale_x(tput):.1f}" cy="{scale_y(inter):.1f}" r="5.5" '
            'fill="#1d4ed8" stroke="#ffffff" stroke-width="1.5">'
            f"<title>{_escape(f_tip)}</title></circle>"
        )

    for rank, entry in enumerate(top_k, 1):
        pt = _as_mapping(entry.get("point"))
        if not pt:
            continue
        cx_k = scale_x(_as_float(pt.get("token_throughput_per_chip")))
        cy_k = scale_y(_as_float(pt.get("est_interactivity_tok_s")))
        svg_parts.extend(
            [
                f'<circle cx="{cx_k:.1f}" cy="{cy_k:.1f}" r="8.5" '
                'fill="none" stroke="#10b981" stroke-width="2.0"/>',
                f'<text x="{cx_k + 10:.1f}" y="{cy_k - 6:.1f}" '
                'font-size="10" font-weight="700" fill="#065f46">'
                f"#{rank}</text>",
            ]
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _render_limits_svg(floor: Mapping[str, object] | None) -> str:
    """Render horizontal bar chart for decode-step resource times."""
    width, height = 720, 290
    pad_left, pad_right, pad_top = 130, 90, 25

    if not floor:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg">'
            f'<text x="{width // 2}" y="{height // 2}" '
            'text-anchor="middle" fill="#64748b">'
            "No decode floor data available</text></svg>"
        )

    hbm_ms = _as_float(floor.get("hbm_bound_ms", floor.get("hbm_ms")))
    matmul_ms = _as_float(
        floor.get(
            "matmul_bound_ms",
            floor.get("matmul_ms", floor.get("matmul_s", floor.get("mm_s", 0.0))),
        )
    )
    vpu_ms = _as_float(
        floor.get("vpu_bound_ms", floor.get("vpu_ms", floor.get("vpu_s", 0.0)))
    )
    ici_ms = _as_float(floor.get("coll_bound_ms", floor.get("ici_ms")))
    chain_ms = _as_float(floor.get("chain_ms"))
    hop_ms = _as_float(floor.get("hop_latency_ms"))
    hard_floor_ms = _as_float(floor.get("hard_floor_ms"))

    items = [
        ("HBM Bound", hbm_ms, "#3b82f6"),
        ("MatMul Bound", matmul_ms, "#10b981"),
        ("Vector (VPU) Bound", vpu_ms, "#06b6d4"),
        ("Collective Bound", ici_ms, "#f59e0b"),
        ("Op Chain", chain_ms, "#8b5cf6"),
        ("Hop Latency", hop_ms, "#ec4899"),
        ("Hard Floor", hard_floor_ms, "#ef4444"),
    ]
    max_val = (max((val for _, val, _ in items), default=1.0) or 1.0) * 1.25
    plot_w = width - pad_left - pad_right
    bar_h, bar_gap = 22, 12

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]
    for i, (label, val, color) in enumerate(items):
        y = pad_top + i * (bar_h + bar_gap)
        bar_w = max(2.0, (val / max_val) * plot_w)
        weight = "bold" if label == "Hard Floor" else "normal"
        svg_parts.extend(
            [
                f'<text x="{pad_left - 12}" y="{y + 15}" text-anchor="end" '
                f'font-size="12" font-weight="{weight}" fill="#334155">'
                f"{_escape(label)}</text>",
                f'<rect x="{pad_left}" y="{y}" width="{bar_w:.1f}" '
                f'height="{bar_h}" fill="{color}" rx="3">'
                f"<title>{_escape(label)}: {val:.3f} ms</title></rect>",
                f'<text x="{pad_left + bar_w + 8}" y="{y + 15}" font-size="11" '
                f'font-weight="{weight}" fill="#475569">{val:.3f} ms</text>',
            ]
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _render_memory_bar(
    pool: Mapping[str, object],
    y: int,
    plot_cfg: tuple[int, int, int],
    default_limit_gb: float,
) -> list[str]:
    """Render one stacked horizontal HBM bar row for a pool."""
    pad_left, plot_w, bar_h = plot_cfg
    role = str(pool.get("role") or "pool")
    topo = str(pool.get("topology") or "")
    limit_gb = _as_float(pool.get("hbm_limit_gb"), default_limit_gb)
    if limit_gb <= 0.0:
        limit_gb = default_limit_gb
    w_gb = _as_float(pool.get("hbm_weights_gb"))
    kv_gb = _as_float(pool.get("hbm_kv_gb"))
    st_gb = _as_float(pool.get("hbm_state_gb"))
    used_gb = _as_float(pool.get("hbm_used_gb"), w_gb + kv_gb + st_gb)
    free_gb = max(0.0, limit_gb - used_gb)

    pool_label = _escape(f"{role.capitalize()} ({topo})")
    parts = [
        f'<text x="{pad_left - 12}" y="{y + 20}" text-anchor="end" '
        f'font-size="12" fill="#334155">{pool_label}</text>'
    ]
    segments = [
        (w_gb, "#3b82f6", f"Weights: {w_gb:.2f} GB"),
        (kv_gb, "#10b981", f"KV Cache: {kv_gb:.2f} GB"),
        (st_gb, "#f59e0b", f"Recurrent State: {st_gb:.2f} GB"),
        (free_gb, "#e2e8f0", f"Headroom: {free_gb:.2f} GB"),
    ]
    cur_x = float(pad_left)
    for gb_val, fill, title in segments:
        seg_w = (gb_val / limit_gb) * plot_w
        if seg_w <= 0.0:
            continue
        if fill == "#e2e8f0":
            seg_w = max(0.0, float(pad_left + plot_w) - cur_x)
        parts.append(
            f'<rect x="{cur_x:.1f}" y="{y}" width="{seg_w:.1f}" '
            f'height="{bar_h}" fill="{fill}"><title>{_escape(title)}</title></rect>'
        )
        cur_x += seg_w

    util_pct = (used_gb / limit_gb) * 100.0 if limit_gb > 0.0 else 0.0
    parts.extend(
        [
            f'<rect x="{pad_left}" y="{y}" width="{plot_w}" height="{bar_h}" '
            'fill="none" stroke="#94a3b8" rx="3"/>',
            f'<text x="{pad_left + plot_w + 10}" y="{y + 17}" font-size="11" '
            f'fill="#475569">{used_gb:.1f} / {limit_gb:.1f} GB '
            f"({util_pct:.0f}%)</text>",
        ]
    )
    return parts


def _render_memory_svg(
    pools: Sequence[Mapping[str, object]],
    default_limit_gb: float,
) -> str:
    """Render inline stacked horizontal bar chart for HBM footprint vs capacity."""
    width, row_h = 720, 60
    pad_left, pad_right, pad_top, bar_h = 130, 110, 25, 24
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
        svg_parts.extend(
            _render_memory_bar(pool, y, (pad_left, plot_w, bar_h), default_limit_gb)
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
.container { max-width: 1600px; margin: 0 auto; }
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
.badge-fact { background-color: #dcfce7; color: #166534; }
.badge-derived { background-color: #e0f2fe; color: #0369a1; }
.badge-assumed { background-color: #fef3c7; color: #92400e; }
.badge-config { background-color: #f3e8ff; color: #6b21a8; }
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  overflow-x: auto;
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
  max-width: 280px;
  white-space: pre-wrap;
  word-break: break-word;
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


def _downsample_embedded_candidates(
    raw_cands: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Downsample raw candidates list when exceeding maximum embedded budget."""
    if len(raw_cands) <= MAX_EMBEDDED_CANDIDATES:
        return list(raw_cands)
    stride = math.ceil(len(raw_cands) / MAX_EMBEDDED_CANDIDATES)
    return list(raw_cands[::stride][:MAX_EMBEDDED_CANDIDATES])


def _extract_recommended_view(
    rec: Mapping[str, object] | None,
) -> tuple[dict[str, object], list[Mapping[str, object]], str]:
    """Extract display metrics, instance pools, and summary for recommended."""
    if rec is not None and bool(rec.get("slo_met", True)):
        pools_list: list[Mapping[str, object]] = []
        p_pool = _as_mapping(rec.get("prefill_pool"))
        d_pool = _as_mapping(rec.get("decode_pool"))
        if p_pool:
            pools_list.append(p_pool)
        if d_pool:
            pools_list.append(d_pool)
        pools_summary = " + ".join(
            f"{p.get('role', '')} {pool_label(p)}" for p in pools_list
        )
        floor = _as_mapping(rec.get("decode_floor"))
        ttft_val = _as_float(rec.get("est_ttft_ms"))
        tpot_val = _as_float(rec.get("est_tpot_ms"))
        bound_ttft = rec.get("bound_ttft_ms")
        bound_tpot = rec.get("bound_tpot_ms")
        rec_fields: dict[str, object] = {
            "mode": str(rec.get("mode", "")).upper(),
            "chips": _as_int(rec.get("total_chips"), 0),
            "concurrency": _as_int(rec.get("concurrency"), 1),
            "tput": f"{_as_float(rec.get('token_throughput_per_chip')):.1f}",
            "out_tput": f"{_as_float(rec.get('output_throughput_per_chip')):.1f}",
            "interactivity": f"{_as_float(rec.get('est_interactivity_tok_s')):.1f}",
            "ttft": (
                f"{_as_float(bound_ttft):.1f} ~ {ttft_val:.1f}"
                if bound_ttft is not None
                else f"{ttft_val:.1f}"
            ),
            "tpot": (
                f"{_as_float(bound_tpot):.2f} ~ {tpot_val:.2f}"
                if bound_tpot is not None
                else f"{tpot_val:.2f}"
            ),
            "binding": str(rec.get("binding", "hbm")),
            "decode_floor": floor or None,
        }
        return rec_fields, pools_list, pools_summary

    return (
        {
            "mode": "INFEASIBLE",
            "chips": 0,
            "concurrency": 0,
            "tput": "0.0",
            "out_tput": "0.0",
            "interactivity": "0.0",
            "ttft": "-",
            "tpot": "-",
            "binding": "None",
            "decode_floor": None,
        },
        [],
        "No feasible layout met constraints",
    )


def _render_warnings_html(warnings: Sequence[str]) -> str:
    """Format warning messages into an alert box HTML element."""
    if not warnings:
        return ""
    warn_items = "".join(f"<li>{_escape(w)}</li>" for w in warnings)
    return (
        '<div class="alert-box alert-warning">'
        f"<strong>Warnings:</strong><ul>{warn_items}</ul></div>"
    )


def _format_launch_lines(commands: Sequence[object]) -> str:
    """Join launch command lines cleanly on newlines so comments never tail commands."""
    return "\n".join(str(cmd) for cmd in commands if str(cmd).strip())


def _render_launch_cmds_html(top_k: Sequence[Mapping[str, object]]) -> str:
    """Format recommended launch command into a code box HTML element."""
    if not top_k:
        return ""
    launch = _as_mapping(top_k[0].get("launch"))
    cmds = _as_sequence(launch.get("commands"))
    cmds_str = _format_launch_lines(cmds)
    if not cmds_str:
        return ""
    return (
        '<div class="code-box">'
        '<div class="code-header">Recommended Launch Command</div>'
        f"<pre><code>{_escape(cmds_str)}</code></pre></div>"
    )


def _render_table_rows(
    top_k: Sequence[Mapping[str, object]],
    reference: Mapping[str, object] | None = None,
) -> str:
    """Format top candidate layout options into HTML table rows."""
    if not top_k:
        return (
            '<tr><td colspan="13" style="text-align:center;">'
            "No feasible layouts found</td></tr>"
        )
    entries: list[Mapping[str, object]] = list(top_k)
    if (
        reference is not None
        and bool(reference.get("found"))
        and not bool(reference.get("in_top_k"))
        and _as_mapping(reference.get("point"))
    ):
        entries.append(
            {
                "frontier_role": "reference",
                "point": _as_mapping(reference.get("point")),
                "launch": _as_mapping(reference.get("launch")),
            }
        )

    rows: list[str] = []
    for rank, entry in enumerate(entries, 1):
        candidate_point = _as_mapping(entry.get("point"))
        launch = _as_mapping(entry.get("launch"))
        frontier_role = str(entry.get("frontier_role", "candidate"))
        rank_label = f"#{rank}" if frontier_role != "reference" else "-"
        serving_mode = str(candidate_point.get("mode", ""))
        concurrency = _as_int(candidate_point.get("concurrency"), 1)
        total_chips = _as_int(candidate_point.get("total_chips"), 0)
        interactivity_str = (
            f"{_as_float(candidate_point.get('est_interactivity_tok_s')):.1f}"
        )
        throughput_str = (
            f"{_as_float(candidate_point.get('token_throughput_per_chip')):.1f}"
        )
        bound_tpot_str = f"{_as_float(candidate_point.get('bound_tpot_ms')):.2f}"
        est_ttft_str = f"{_as_float(candidate_point.get('est_ttft_ms')):.1f}"
        est_tpot_str = f"{_as_float(candidate_point.get('est_tpot_ms')):.2f}"
        capacity_raw = candidate_point.get("kv_cache_capacity_tokens")
        capacity_str = (
            f"{_as_int(capacity_raw):,}" if capacity_raw is not None else "inf"
        )
        effective_hit_rate = _as_float(candidate_point.get("effective_prefix_hit_rate"))
        launch_command_str = _format_launch_lines(_as_sequence(launch.get("commands")))

        host_count = max(1, total_chips // DEFAULT_CHIPS_PER_HOST)
        core_count = total_chips * DEFAULT_CORES_PER_CHIP
        host_suffix = "host" if host_count == 1 else "hosts"
        chips_description = (
            f"{total_chips} chips ({host_count} {host_suffix}, {core_count} cores)"
        )

        pool_summaries: list[str] = []
        prefill_pool = _as_mapping(candidate_point.get("prefill_pool"))
        decode_pool = _as_mapping(candidate_point.get("decode_pool"))
        if prefill_pool:
            prefill_instances = _as_int(prefill_pool.get("instances"), 1)
            prefill_chips = (
                _as_int(prefill_pool.get("chips_per_instance"), 0) * prefill_instances
            )
            prefill_hosts = max(1, prefill_chips // DEFAULT_CHIPS_PER_HOST)
            prefill_cores = prefill_chips * DEFAULT_CORES_PER_CHIP
            h_suffix = "s" if prefill_hosts > 1 else ""
            prefill_pool_desc = (
                f" ({prefill_instances} inst, {prefill_chips} chips, "
                f"{prefill_hosts} host{h_suffix}, {prefill_cores} cores)"
            )
            pool_summaries.append(f"P {pool_label(prefill_pool)}{prefill_pool_desc}")
        if decode_pool:
            prefix = "D " if prefill_pool else ""
            decode_instances = _as_int(decode_pool.get("instances"), 1)
            decode_chips = (
                _as_int(decode_pool.get("chips_per_instance"), 0) * decode_instances
            )
            decode_hosts = max(1, decode_chips // DEFAULT_CHIPS_PER_HOST)
            decode_cores = decode_chips * DEFAULT_CORES_PER_CHIP
            h_suffix = "s" if decode_hosts > 1 else ""
            decode_pool_desc = (
                f" ({decode_instances} inst, {decode_chips} chips, "
                f"{decode_hosts} host{h_suffix}, {decode_cores} cores)"
            )
            pool_summaries.append(
                f"{prefix}{pool_label(decode_pool)}{decode_pool_desc}"
            )
        pools_display = " + ".join(pool_summaries)

        rows.append(
            f"<tr>"
            f"<td>{rank_label}</td>"
            f"<td><code>{_escape(frontier_role)}</code></td>"
            f"<td><strong>{_escape(serving_mode)}</strong></td>"
            f"<td>{concurrency}</td>"
            f"<td>{chips_description}</td>"
            f"<td><code>{_escape(pools_display)}</code></td>"
            f"<td>{interactivity_str}</td>"
            f"<td>{throughput_str}</td>"
            f"<td>{bound_tpot_str}</td>"
            f"<td>{est_ttft_str}</td>"
            f"<td>{est_tpot_str}</td>"
            f"<td>{_escape(capacity_str)} / {effective_hit_rate:.2f}</td>"
            f'<td><code class="cmd-cell" title="{_escape(launch_command_str)}">'
            f"{_escape(launch_command_str)}</code></td>"
            f"</tr>"
        )
    return "".join(rows)


def _derive_default_hbm_limit_gb(payload: Mapping[str, object]) -> float:
    """Derive per-core HBM limit (GB) from payload chip_spec and utilization."""
    chip_spec = _as_mapping(payload.get("chip_spec"))
    util = _as_float(
        payload.get("gpu_memory_utilization"),
        DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    fallback_chip = CHIPS["tpu7x"]
    hbm_bytes = _as_float(chip_spec.get("hbm_bytes"), float(fallback_chip.hbm_bytes))
    return (hbm_bytes * util) / UNIT_GIGA


def _build_dynamic_chip_spec(
    chip_spec: Mapping[str, object],
    fallback: ChipSpec,
    chip_name: str,
) -> ChipSpec:
    """Build ChipSpec when payload lacks an embedded catalog."""
    return ChipSpec(
        name=chip_name,
        hbm_bytes=int(_as_float(chip_spec.get("hbm_bytes"), float(fallback.hbm_bytes))),
        hbm_bw=_as_float(chip_spec.get("hbm_bw"), fallback.hbm_bw),
        bf16_flops=_as_float(chip_spec.get("bf16_flops"), fallback.bf16_flops),
        fp8_flops=_as_float(chip_spec.get("fp8_flops"), fallback.fp8_flops),
        vmem_bytes=int(
            _as_float(chip_spec.get("vmem_bytes"), float(fallback.vmem_bytes))
        ),
        ici_bw=_as_float(chip_spec.get("ici_bw"), fallback.ici_bw),
        cores_per_chip=_as_int(
            chip_spec.get("cores_per_chip"), fallback.cores_per_chip
        ),
        ici_link_bw=_as_float(chip_spec.get("ici_link_bw"), fallback.ici_link_bw),
        d2d_bw=_as_float(chip_spec.get("d2d_bw"), fallback.d2d_bw),
        ici_hop_latency_s=_as_float(
            chip_spec.get("ici_hop_latency_s"), fallback.ici_hop_latency_s
        ),
        mxu_m_tile=_as_int(chip_spec.get("mxu_m_tile"), fallback.mxu_m_tile),
        dcn_host_gbit_per_s=_as_float(
            chip_spec.get("dcn_host_gbit_per_s"), fallback.dcn_host_gbit_per_s
        ),
        chips_per_host=_as_int(
            chip_spec.get("chips_per_host"), fallback.chips_per_host
        ),
        dcn_bw=_as_float(chip_spec.get("dcn_bw"), fallback.dcn_bw),
        pcie_bw=_as_float(chip_spec.get("pcie_bw"), fallback.pcie_bw),
    )


def _format_catalog_entry_row(key: str, entry: Mapping[str, object]) -> str:
    """Render one catalog entry into an HTML table row."""
    kind = _escape(entry.get("kind", ""))
    scope = _escape(entry.get("scope", ""))
    unit = str(entry.get("unit", ""))
    val = entry.get("value", "")
    val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
    desc_str = _escape(entry.get("description", ""))
    src = _escape(entry.get("source", ""))
    display = entry.get("display")
    computed = (
        format_constant_display(_as_float(val), unit)
        if (val != "" and unit != "")
        else ""
    )
    if not display or (
        computed and str(computed).split()[0] != str(display).split()[0]
    ):
        display = computed or display
    display_str = _escape(display or f"{val_str} {unit}")
    val_sub = f"{val} {unit}".strip()
    return (
        "          <tr>"
        f"<td><code>{_escape(key)}</code></td>"
        f"<td><strong>{display_str}</strong><br>"
        f'<small style="color: #64748b;">{_escape(val_sub)}</small></td>'
        f"<td><code>{scope}</code></td>"
        f'<td><span class="badge badge-{kind}">{kind}</span></td>'
        f"<td>{desc_str}</td>"
        f'<td style="font-size: 0.85em; color: #475569;">{src}</td>'
        "</tr>"
    )


def _render_chip_specs_item(payload: Mapping[str, object]) -> str:
    """Render dynamic Per-Core hardware specification line from chip_spec."""
    fallback = CHIPS["tpu7x"]
    chip_spec = _as_mapping(payload.get("chip_spec"))
    util = _as_float(
        payload.get("gpu_memory_utilization"),
        DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    chip_name = str(chip_spec.get("name") or payload.get("chip") or fallback.name)
    hbm_gb = (
        _as_float(chip_spec.get("hbm_bytes"), float(fallback.hbm_bytes)) / UNIT_GIGA
    )
    usable_gb = hbm_gb * util
    hbm_tbs = _as_float(chip_spec.get("hbm_bw"), fallback.hbm_bw) / UNIT_TERA
    bf16_pf = _as_float(chip_spec.get("bf16_flops"), fallback.bf16_flops) / UNIT_PETA
    fp8_pf = _as_float(chip_spec.get("fp8_flops"), fallback.fp8_flops) / UNIT_PETA
    ici_link_gbps = (
        _as_float(chip_spec.get("ici_link_bw"), fallback.ici_link_bw) / UNIT_GIGA
    )
    ici_bw_gbps = _as_float(chip_spec.get("ici_bw"), fallback.ici_bw) / UNIT_GIGA
    d2d_gbps = _as_float(chip_spec.get("d2d_bw"), fallback.d2d_bw) / UNIT_GIGA
    hop_us = (
        _as_float(chip_spec.get("ici_hop_latency_s"), fallback.ici_hop_latency_s)
        * UNIT_MEGA
    )
    dcn_gbps = _as_float(chip_spec.get("dcn_bw"), fallback.dcn_bw) / UNIT_GIGA
    pcie_gbps = _as_float(chip_spec.get("pcie_bw"), fallback.pcie_bw) / UNIT_GIGA
    m_tile = _as_int(chip_spec.get("mxu_m_tile"), 0)
    tile_str = f"<code>{m_tile}</code>-row systolic matrix tile, " if m_tile > 0 else ""
    return (
        f"      <li><strong>{_escape(chip_name)} Per-Core Specs:</strong> "
        f"<code>{hbm_gb:.1f} GB</code> HBM "
        f"(<code>{usable_gb:.2f} GB</code> at <code>{util:.2f}</code> utilization), "
        f"<code>{hbm_tbs:.2f} TB/s</code> HBM, "
        f"<code>{bf16_pf:.3f} PFLOP/s</code> BF16, "
        f"<code>{fp8_pf:.3f} PFLOP/s</code> FP8, "
        f"{tile_str}"
        f"<code>{ici_link_gbps:.0f} GB/s</code>/link "
        f"(<code>{ici_bw_gbps:.0f} GB/s</code>/core, "
        f"<code>{d2d_gbps:.0f} GB/s</code>/chip D2D) ICI + "
        f"<code>{hop_us:.1f} us</code> hop, "
        f"<code>{dcn_gbps:.2f} GB/s</code>/chip DCN, and "
        f"<code>{pcie_gbps:.1f} GB/s</code>/chip PCIe.</li>"
    )


def _render_assumptions_card(payload: Mapping[str, object]) -> list[str]:
    """Render Assumptions, Hardware Specs & Scope card with constants catalog."""
    fallback = CHIPS["tpu7x"]
    chip_spec = _as_mapping(payload.get("chip_spec"))
    chip_name = str(chip_spec.get("name") or payload.get("chip") or fallback.name)
    cat_raw = chip_spec.get("catalog")
    if isinstance(cat_raw, Mapping):
        cat = cat_raw
    else:
        cat = chip_catalog(_build_dynamic_chip_spec(chip_spec, fallback, chip_name))
    cat_rows = [
        _format_catalog_entry_row(str(k), _as_mapping(entry_raw))
        for k, entry_raw in cat.items()
        if isinstance(entry_raw, Mapping)
    ]
    chip_specs_item = _render_chip_specs_item(payload)
    return [
        '  <section class="card assumptions-card">',
        "    <h2>Assumptions, Hardware Specs &amp; Scope</h2>",
        "    <ul>",
        "      <li><strong>Datasheet Bounds:</strong> <code>Hard Floor</code> is "
        "<code>max(HBM, Matmul, Vector, ICI)</code> "
        "(full communication-compute overlap); "
        "<code>Serial Estimate</code> sums per-module "
        "<code>max(HBM, Matmul, Vector)</code> plus exposed <code>ICI Chain</code> "
        "(zero overlap) for layout ranking.</li>",
        chip_specs_item,
        "      <li><strong>Intentionally Omitted (Day-1 Trace Scope):</strong> "
        "Empirical kernel utilization derates, XLA temporary workspace "
        "(<code>8 ~ 15 GB/core</code>), "
        "and trace-level LRU cache eviction or arrival burstiness.</li>",
        "    </ul>",
        "    <details open><summary>Constants Catalog "
        "(fact / derived / assumed / config)</summary>",
        '    <div style="overflow-x: auto; margin-top: 0.75rem;">',
        "      <table>",
        "        <thead>",
        "          <tr>",
        '            <th style="text-align: left;">Constant</th>',
        '            <th style="text-align: left;">Value</th>',
        '            <th style="text-align: left;">Scope</th>',
        '            <th style="text-align: left;">Kind</th>',
        '            <th style="text-align: left;">Description</th>',
        '            <th style="text-align: left;">Source / Derivation</th>',
        "          </tr>",
        "        </thead>",
        "        <tbody>",
        *[f"    {row}" for row in cat_rows],
        "        </tbody>",
        "      </table>",
        "    </div>",
        "    </details>",
        "  </section>",
    ]


def _render_crossover_stat_html(payload: Mapping[str, object]) -> str:
    """Render prefix-hit regime banner box when sweep winner range exists."""
    win_range = _as_sequence(payload.get("winner_prefix_hit_range"))
    if len(win_range) != PREFIX_RANGE_PAIR_LEN:
        return ""
    cross_rate = payload.get("prefix_hit_crossover_rate")
    range_txt = f"[{_as_float(win_range[0]):.2f}, {_as_float(win_range[1]):.2f}]"
    cross_txt = (
        f"crossover={_as_float(cross_rate):.2f}" if cross_rate is not None else "stable"
    )
    return (
        '      <div class="stat-box">'
        '        <div class="stat-label">Prefix-Hit Regime</div>'
        '        <div class="stat-value" style="font-size:15px;">'
        f"{_escape(range_txt)} "
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        f"({_escape(cross_txt)})</span></div>"
        "      </div>"
    )


def render_html(payload: Mapping[str, object]) -> str:  # noqa: PLR0915
    """Render a fully self-contained, offline HTML report from a solve/v1 payload."""
    rec = _as_mapping(payload.get("recommended")) or None
    frontier = [_as_mapping(x) for x in _as_sequence(payload.get("frontier"))]
    candidates = [_as_mapping(x) for x in _as_sequence(payload.get("candidates"))]
    search = _as_mapping(payload.get("search"))
    scenario = _as_mapping(payload.get("scenario"))
    warnings = [str(w) for w in _as_sequence(payload.get("warnings"))]
    top_k = [_as_mapping(x) for x in _as_sequence(payload.get("top_k"))]
    model = str(payload.get("model") or "Model")
    chip = str(payload.get("chip") or "TPU")

    embedded_payload = dict(payload)
    embedded_payload.pop("sweep_candidates", None)
    embedded_payload["candidates"] = _downsample_embedded_candidates(candidates)
    embedded_json = json.dumps(embedded_payload, separators=(",", ":")).replace(
        "</", "<\\/"
    )

    if rec is not None and not bool(rec.get("slo_met", True)) and not warnings:
        warnings = [
            "No candidate layout met latency SLOs; recommended layout is infeasible.",
        ]

    rec_fields, pools_list, pools_summary = _extract_recommended_view(rec)
    decode_floor_raw = rec_fields["decode_floor"]
    decode_floor = _as_mapping(decode_floor_raw) if decode_floor_raw else None
    default_limit_gb = _derive_default_hbm_limit_gb(payload)

    frontier_svg = _render_frontier_svg(candidates, frontier, top_k=top_k)
    limits_svg = _render_limits_svg(decode_floor)
    memory_svg = _render_memory_svg(pools_list, default_limit_gb)

    enum_cnt = _as_int(search.get("enumerated"))
    hbm_cnt = _as_int(search.get("hbm_fit"))
    slo_cnt = _as_int(search.get("slo_met"))
    front_cnt = _as_int(search.get("frontier"))

    warnings_html = _render_warnings_html(warnings)
    launch_cmds_html = _render_launch_cmds_html(top_k)
    ref_report = _as_mapping(payload.get("reference"))
    top_table_body = _render_table_rows(top_k, reference=ref_report)
    crossover_stat_html = _render_crossover_stat_html(payload)

    scen_name = _escape(scenario.get("name", "custom"))
    scen_isl = _as_int(scenario.get("isl"))
    scen_osl = _as_int(scenario.get("osl"))
    scen_hit = _as_float(scenario.get("prefix_hit_rate"))
    chip_budget = _as_mapping(payload.get("chip_budget")) or {}
    budget_chips = _as_int(chip_budget.get("max_chips", payload.get("budget", 0)))
    budget_src = str(
        chip_budget.get("budget_source", payload.get("budget_source", "flag"))
    )
    budget_hosts = max(1, budget_chips // DEFAULT_CHIPS_PER_HOST)
    budget_str = (
        f"Search Budget: {budget_chips} chips / {budget_hosts} hosts ({budget_src})"
        if budget_chips > 0
        else ""
    )
    dtypes_map = _as_mapping(payload.get("dtypes")) or {}
    if dtypes_map:
        w_dense_val = dtypes_map.get("w_dense", 0.0)
        w_dense_dt = dtypes_map.get("w_dense_dtype", "dense")
        w_dense_src = dtypes_map.get("w_dense_source", "default")
        w_expert_val = dtypes_map.get("w_expert", 0.0)
        w_expert_dt = dtypes_map.get("w_expert_dtype", "expert")
        w_expert_src = dtypes_map.get("w_expert_source", "default")
        kv_val = dtypes_map.get("kv", 0.0)
        kv_dt = dtypes_map.get("kv_dtype", "kv")
        kv_src = dtypes_map.get("kv_source", "default")
        act_val = dtypes_map.get("act", "?")
        w_d_s = (
            f"{w_dense_val:.4g}B"
            if isinstance(w_dense_val, (int, float))
            else f"{w_dense_val}B"
        )
        w_e_s = (
            f"{w_expert_val:.4g}B"
            if isinstance(w_expert_val, (int, float))
            else f"{w_expert_val}B"
        )
        kv_s = f"{kv_val:.4g}B" if isinstance(kv_val, (int, float)) else f"{kv_val}B"
        dtypes_str = (
            f"Dtypes: dense={w_dense_dt} ({w_d_s}, {w_dense_src}), "
            f"expert={w_expert_dt} ({w_e_s}, {w_expert_src}), "
            f"kv={kv_dt} ({kv_s}, {kv_src}), act={act_val}"
        )
    else:
        dtypes_str = ""
    pins = _as_mapping(chip_budget.get("pins", payload.get("pins"))) or {}
    pin_parts: list[str] = []
    if pins.get("prefill_topology"):
        pin_parts.append(f"P-slice: {pins['prefill_topology']}")
    if pins.get("decode_topology"):
        pin_parts.append(f"D-slice: {pins['decode_topology']}")
    pd_ratio_raw = pins.get("pd_ratio")
    if (
        isinstance(pd_ratio_raw, Sequence)
        and not isinstance(pd_ratio_raw, (str, bytes))
        and len(pd_ratio_raw) >= PREFIX_RANGE_PAIR_LEN
    ):
        pin_parts.append(f"P:D={pd_ratio_raw[0]}:{pd_ratio_raw[1]}")
    pin_str = f" [Pins: {', '.join(pin_parts)}]" if pin_parts else ""

    sh_fit = _as_mapping(payload.get("single_host_fit"))
    sh_fit_html = ""
    if sh_fit:
        sh_feas = bool(sh_fit.get("feasible", False))
        sh_status = str(sh_fit.get("status", "REJECTED"))
        sh_reason = str(sh_fit.get("reason", ""))
        sh_used = _as_float(sh_fit.get("hbm_used_gb"))
        sh_lim = _as_float(sh_fit.get("hbm_limit_gb"))
        sh_par = str(sh_fit.get("best_parallelism", "TP8"))
        bg_color = "#dcfce7" if sh_feas else "#fee2e2"
        text_color = "#166534" if sh_feas else "#991b1b"
        sh_fit_html = (
            '  <section id="single-host-fit" class="card">\n'
            "    <h2>Single-Host 2x2x1 Fit Verdict "
            "(1 host, 4 chips, 8 TensorCores)</h2>\n"
            '    <div style="font-size:14px; margin-top:8px;">\n'
            f'      <span class="badge" style="background:{bg_color}; '
            f'color:{text_color}; font-weight:bold;">{_escape(sh_status)}</span> '
            "&bull; \n"
            f"      <strong>{_escape(sh_reason)}</strong> &bull; \n"
            f"      HBM Footprint: <code>{sh_used:.2f} GB</code> / "
            f"<code>{sh_lim:.2f} GB</code> limit &bull; \n"
            f"      Best Parallelism: <code>{_escape(sh_par)}</code>\n"
            "    </div>\n"
            "  </section>"
        )

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
        f"(ISL={scen_isl}, OSL={scen_osl}, hit_rate={scen_hit:.2f})"
        + (
            f" &bull; <strong>{_escape(budget_str)}{_escape(pin_str)}</strong>"
            if budget_str
            else ""
        )
        + (f" &bull; <span>{_escape(dtypes_str)}</span>" if dtypes_str else ""),
        "    </div>",
        "  </header>",
        f"  {warnings_html}",
        '  <section id="executive-summary" class="card">',
        "    <h2>Executive Recommendation</h2>",
        "    <div>",
        '      <span class="badge" style="background:#dbeafe; color:#1e40af; '
        f'font-size:13px; padding:4px 10px;">{_escape(rec_fields["mode"])}</span>',
        f"      <strong>{_escape(pools_summary)}</strong> &bull; "
        f"Concurrency {rec_fields['concurrency']} across "
        f"{rec_fields['chips']} chips "
        f"({max(1, _as_int(rec_fields['chips']) // DEFAULT_CHIPS_PER_HOST)} hosts)",
        "    </div>",
        '    <div class="grid-stats">',
        '      <div class="stat-box">',
        '        <div class="stat-label">Throughput / Chip</div>',
        f'        <div class="stat-value">{rec_fields["tput"]} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        f"tok/s/chip (out: {rec_fields['out_tput']})</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">Est. Interactivity</div>',
        f'        <div class="stat-value">{rec_fields["interactivity"]} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "tok/s</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">TTFT (bound ~ est)</div>',
        f'        <div class="stat-value">{rec_fields["ttft"]} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "ms</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">TPOT (bound ~ est)</div>',
        f'        <div class="stat-value">{rec_fields["tpot"]} '
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        "ms</span></div>",
        "      </div>",
        '      <div class="stat-box">',
        '        <div class="stat-label">Binding Unit</div>',
        '        <div class="stat-value" style="font-size:16px;">'
        f"{_escape(rec_fields['binding'])}</div>",
        "      </div>",
        crossover_stat_html,
        "    </div>",
        f"    {launch_cmds_html}",
        "  </section>",
        sh_fit_html,
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
        (
            '    <p class="section-note" '
            'style="color:var(--text-muted);font-size:13px;margin-top:4px;">'
            "Throughput and interactivity are model estimates "
            "(<code>est_*</code>); latency bounds (<code>bound_*</code>) are "
            "hard physical lower limits.</p>"
        ),
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
        "          <th>Concurrency</th>",
        "          <th>Chips</th>",
        "          <th>Pools</th>",
        "          <th>Int (tok/s) [est]</th>",
        "          <th>Tput (tok/s/chip)</th>",
        "          <th>TPOT Bound (ms) [bound]</th>",
        "          <th>TTFT (ms) [est]</th>",
        "          <th>TPOT (ms) [est]</th>",
        "          <th>KV Cap / Prefix Hit Rate</th>",
        "          <th>Launch Command</th>",
        "        </tr>",
        "      </thead>",
        "      <tbody>",
        f"        {top_table_body}",
        "      </tbody>",
        "    </table>",
        *(
            (
                f'    <p class="section-note" '
                f'style="color:var(--text-muted);font-size:13px;margin-top:8px;">'
                f"# reference layout {_escape(str(ref_report.get('layout', '')))}: "
                f"not enumerated "
                f"({_escape(str(ref_report.get('rejected_by', '')))})</p>",
            )
            if (ref_report and not bool(ref_report.get("found")))
            else ()
        ),
        (
            '    <p class="section-note" '
            'style="color:var(--text-muted);font-size:13px;margin-top:8px;">'
            "Note: Columns marked <code>[bound]</code> (such as "
            "<code>bound_tpot_ms</code>) are hard physical lower bounds; "
            "columns marked <code>[est]</code> (such as <code>est_ttft_ms</code> "
            "and <code>est_tpot_ms</code>) are model estimates.</p>"
        ),
        (
            '    <div class="legend-card" style="margin-top:12px; font-size:12px; '
            'color:var(--text-muted); line-height:1.5;">'
            "<strong>Glossary of Physical &amp; Logical Units:</strong>"
            "<ul>"
            "<li><strong>Core (TensorCore):</strong> Physical compute engine "
            "and systolic array execution unit per public Cloud TPU "
            "documentation (Cloud TPU v7x). On TPU v7x, each physical chip "
            "contains 2 TensorCores.</li>"
            "<li><strong>Chip:</strong> Physical TPU package containing 2 TensorCores, "
            "local high-bandwidth memory (HBM), and inter-core interconnect "
            "(ICI) router links.</li>"
            "<li><strong>Host:</strong> Physical host CPU machine connected "
            "to a tray of TPU chips (on TPU v7x, each host controls 4 physical "
            "chips = 8 TensorCores).</li>"
            "<li><strong>Slice:</strong> Dedicated 2D/3D torus mesh of TPU chips "
            "interconnected by low-latency optical/copper ICI links "
            "(e.g. <code>2x2x1</code> = 4 chips / 1 host, "
            "<code>2x2x2</code> = 8 chips / 2 hosts).</li>"
            "<li><strong>Instance:</strong> Independent end-to-end vLLM "
            "serving process replica allocated to a slice (the prefix "
            "<code>1x</code> or <code>4x</code> counts instances).</li>"
            "</ul>"
            "<strong>Metric &amp; Axis Legend:</strong>"
            "<ul>"
            "<li><strong>Concurrency:</strong> Total active in-flight request "
            "streams across serving instances.</li>"
            "<li><strong>Total Chips:</strong> Physical TPU chip count allocated "
            "across prefill and decode pools.</li>"
            "<li><strong>Pools:</strong> Topology slice allocation "
            "(<code>{instances}x {topology} [axes]</code>).</li>"
            "<li><strong>Interactivity (tok/s) [est]:</strong> Estimated token "
            "generation speed per client stream.</li>"
            "<li><strong>Throughput (tok/s/chip):</strong> Total aggregate token "
            "processing throughput per physical chip.</li>"
            "<li><strong>TPOT Bound / Est (ms):</strong> Time-per-output-token "
            "physical lower bound (<code>bound_tpot_ms</code>) vs causally modeled "
            "estimate (<code>est_tpot_ms</code>).</li>"
            "<li><strong>TTFT Est (ms):</strong> Modeled time-to-first-token latency "
            "for chunked prefill.</li>"
            "<li><strong>KV Capacity / Prefix Hit Rate:</strong> "
            "Effective HBM KV cache "
            "capacity (tokens) and prefix-cache hit rate.</li>"
            "<li><strong>Parallelism Axes:</strong> <code>TP</code> (Tensor Parallel), "
            "<code>DP</code> (Data Parallel), <code>PP</code> (Pipeline Parallel), "
            "<code>PCP</code> (Prefill Context Parallel), "
            "<code>DCP</code> (Decode Context Parallel), "
            "<code>EP</code> (Expert Parallel).</li>"
            "<li><strong>Traffic &amp; Generation:</strong> <code>P</code> (Prefill "
            "pool), <code>D</code> (Decode pool), <code>ISL</code> (Input Sequence "
            "Length), <code>OSL</code> (Output Sequence Length), "
            "<code>MTP</code> (Multi-Token Prediction draft layers).</li>"
            "</ul>"
            "</div>"
        ),
        "  </section>",
        *_render_assumptions_card(payload),
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


def write_html(payload: Mapping[str, object], path: Path | str) -> Path:
    """Render and write HTML report to disk at path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_html(payload)
    dest.write_text(rendered, encoding="utf-8")
    return dest
