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
"""Terminal regime crossover table and self-contained inline-SVG HTML report."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vllm_torchtpu.megakernel.advisor import (
        AdvisorReport,
        AdvisorSweepReport,
    )

_MOE_MODE_SHORT_HINTS: Mapping[str, str] = {
    "masked_on_chip": "Routed experts stream into VMEM; token rows are masked "
    "per expert inside the fused call; no reordering",
    "grouped_on_chip": "Tokens are grouped by expert inside VMEM, then each "
    "expert runs on its rows inside the fused call",
    "grouped_in_hbm": "Tokens are sorted by expert in HBM; the MoE is its own "
    "call, so the kernel boundary falls at each MoE",
    "dense_mlp": "Dense feed-forward; no routed experts",
}
_SCOPE_HINTS: Mapping[str, str] = {
    "whole_step": "one call for the whole step; [T, H] stays in VMEM, and tokens "
    "past the VMEM ceiling are tiled inside each layer while its weights stay "
    "resident",
    "whole_step_repass": "one call per token pass past the VMEM ceiling; every "
    "pass re-reads all weights",
    "layer_group": "one call per group of N layers, tiled like whole_step",
    "single_layer": "one call per layer",
    "per_op": "one call per operator; activations hand off through HBM",
}


def _esc(val: object) -> str:
    """Escape a value for safe HTML rendering."""
    return "" if val is None else html.escape(str(val))


def format_advisor_summary(rep: AdvisorReport) -> str:
    """Format single-workload AdvisorReport for terminal output."""
    slope_kib = rep.per_token_vmem_bytes / 1024.0
    ridge_fb = rep.ridge_dense_flop_per_byte
    lines = [
        (
            f"Model: {rep.model} | Chip: {rep.chip} | "
            f"Phase: {rep.phase} (T={rep.active_tokens})"
        ),
        (
            f"Recommended: {rep.recommended_scope} "
            f"[moe={rep.recommended_moe_mode}] "
            f"(build={rep.build_megakernel})"
        ),
        f"Rationale: {rep.rationale}",
        (
            f"VMEM Budget: {rep.vmem_budget_mib:.1f} MiB "
            f"({rep.dma_buffers}-slot async DMA) | "
            f"Per-Token Slope: {slope_kib:.1f} KiB/tok | "
            f"Ridge: {ridge_fb:.1f} FLOP/B (T_ridge={rep.ridge_tokens})"
        ),
    ]
    lines.extend(
        f"  - {sc.scope:12s} {sc.step_bound_us:9.1f} us "
        f"[t_max={sc.t_vmem_max:4d}, passes={sc.n_splits:2d}, "
        f"static_vmem={sc.vmem_static_mib:5.2f} MiB]"
        for sc in rep.scopes
    )
    return "\n".join(lines)


def format_regime_table(sweep: AdvisorSweepReport) -> str:
    """Format multi-regime sweep table and tailoring guide for terminal output."""
    rec = sweep.parallelism
    vmem = sweep.vmem_breakdown_bytes
    per_tok_kib = sweep.per_token_vmem_bytes / 1024.0
    mode_tag = rec.source or (
        "auto-recommended" if rec.auto_selected else "user-specified"
    )
    hdr = (
        f"{'Request Regime':36s} | {'Layout':22s} | {'Tokens':>6s} | "
        f"{'AI (FLOP/B)':>11s} | {'Passes':>6s} | {'Scope':12s} | "
        f"{'MoE Mode':15s} | {'Step (us)':>10s}"
    )
    layout_lines = [
        f"Decode Parallelism: {rec.label()} [{rec.topology}] ({mode_tag})",
    ]
    pre = sweep.prefill_parallelism
    if pre is not None:
        groups = f", {len(pre.layer_groups)} layer groups" if pre.layer_groups else ""
        layout_lines.append(
            f"Prefill Parallelism: {pre.label()} [{pre.topology}] "
            f"({pre.source}{groups})"
        )
    lines = [
        f"=== Megakernel Advisor Regime Report: {sweep.model} on {sweep.chip} ===",
        *layout_lines,
        (
            f"HBM Footprint: {rec.hbm_used_gb:.2f} / {rec.hbm_capacity_gb:.1f} GB/core "
            f"(active weights/step: {rec.w_step_gb:.2f} GB/core)"
        ),
        (
            f"VMEM Ceiling (whole_step, {sweep.dma_buffers}-slot DMA): "
            f"max_vmem_tokens = {sweep.max_vmem_tokens} tokens "
            f"(usable={sweep.vmem_budget_mib:.1f} MiB, "
            f"static={sweep.vmem_static_mib:.2f} MiB, "
            f"slope={per_tok_kib:.1f} KiB/tok "
            f"[residual={vmem['residual_buffer_bytes'] // 1024}K, "
            f"attn={vmem['attn_scratch_bytes'] // 1024}K, "
            f"linear_attn={vmem['linear_attn_scratch_bytes'] // 1024}K, "
            f"moe={vmem['moe_scratch_bytes'] // 1024}K])"
        ),
        (
            f"Roofline Ridge Crossover: {sweep.ridge_dense_flop_per_byte:.1f} FLOP/B "
            f"(T_ridge = {sweep.ridge_tokens} active tokens/step)"
        ),
        "",
        hdr,
        "-" * len(hdr),
    ]
    lines.extend(
        f"{entry.label:36s} | {entry.layout:22s} | {entry.active_tokens:6d} | "
        f"{entry.step_arithmetic_intensity:11.2f} | {entry.num_vmem_splits:6d} | "
        f"{entry.recommended_scope:12s} | {entry.recommended_moe_mode:15s} | "
        f"{entry.step_bound_us:10.1f}"
        for entry in sweep.regimes
    )
    lines.extend(
        [
            "",
            "Scope (where the kernel boundary falls):",
            *(f"  * {name:14s}: {hint}." for name, hint in _SCOPE_HINTS.items()),
            "MoE Mode (where tokens are grouped by expert):",
            *(
                f"  * {name:14s}: {hint}."
                for name, hint in _MOE_MODE_SHORT_HINTS.items()
            ),
            "",
            "Tailoring Guide:",
        ]
    )
    lines.extend(f"  * {rule}" for rule in sweep.tailoring_guide)
    return "\n".join(lines)


def _render_vmem_svg(sweep: AdvisorSweepReport) -> str:
    """Render inline SVG chart of VMEM token budget vs usable VMEM capacity."""
    width, height = 680, 260
    pad_l, pad_r, pad_t, pad_b = 64, 28, 28, 44
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    cap_mib = max(sweep.vmem_budget_mib, 64.0)
    static_vmem_mib = sweep.vmem_static_mib
    residual_per_tok_mib = sweep.vmem_breakdown_bytes["residual_buffer_bytes"] / (
        1 << 20
    )
    per_op_per_tok_mib = (
        sweep.per_token_vmem_bytes - sweep.vmem_breakdown_bytes["residual_buffer_bytes"]
    ) / (1 << 20)

    token_ticks = [4, 8, 16, 64, 128, 256, 512]
    bar_w = pw / (len(token_ticks) * 1.6)
    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
        ),
        (
            f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" '
            'fill="#0f172a" stroke="#334155"/>'
        ),
    ]
    y_cap = pad_t + ph - min(1.0, sweep.vmem_budget_mib / cap_mib) * ph
    parts.append(
        f'<line x1="{pad_l}" y1="{y_cap:.1f}" x2="{pad_l + pw}" y2="{y_cap:.1f}" '
        'stroke="#f43f5e" stroke-width="2" stroke-dasharray="5,4"/>'
    )
    parts.append(
        f'<text x="{pad_l + pw - 6}" y="{y_cap - 6:.1f}" text-anchor="end" '
        'font-size="11" fill="#fda4af">Usable VMEM Ceiling '
        f"({sweep.vmem_budget_mib:.1f} MiB, T_max={sweep.max_vmem_tokens})</text>"
    )

    for idx, tok in enumerate(token_ticks):
        cx = pad_l + (idx + 0.5) * (pw / len(token_ticks))
        x0 = cx - bar_w / 2
        residual_mib = residual_per_tok_mib * tok
        per_op_mib = per_op_per_tok_mib * tok
        total_mib = static_vmem_mib + residual_mib + per_op_mib
        h_static = min(ph, (static_vmem_mib / cap_mib) * ph)
        h_residual = min(ph - h_static, (residual_mib / cap_mib) * ph)
        h_per_op = min(ph - h_static - h_residual, (per_op_mib / cap_mib) * ph)

        y_static = pad_t + ph - h_static
        y_residual = y_static - h_residual
        y_per_op = y_residual - h_per_op
        parts.append(
            f'<rect x="{x0:.1f}" y="{y_static:.1f}" width="{bar_w:.1f}" '
            f'height="{h_static:.1f}" fill="#3b82f6">'
            f"<title>T={tok}: {sweep.dma_buffers}-Slot Async Weight Buffers &amp; "
            f"Scratch = {static_vmem_mib:.2f} MiB</title></rect>"
        )
        parts.append(
            f'<rect x="{x0:.1f}" y="{y_residual:.1f}" width="{bar_w:.1f}" '
            f'height="{h_residual:.1f}" fill="#10b981">'
            f"<title>T={tok}: On-Chip Residual Buffer [T, H] = "
            f"{residual_mib:.2f} MiB</title></rect>"
        )
        parts.append(
            f'<rect x="{x0:.1f}" y="{y_per_op:.1f}" width="{bar_w:.1f}" '
            f'height="{h_per_op:.1f}" fill="#f59e0b">'
            f"<title>T={tok}: Shared Operator Scratchpad = {per_op_mib:.2f} MiB "
            f"(Total={total_mib:.1f} MiB)</title></rect>"
        )
        passes = (
            1
            if sweep.max_vmem_tokens > 0 and tok <= sweep.max_vmem_tokens
            else (
                math.ceil(tok / max(1, sweep.max_vmem_tokens))
                if sweep.max_vmem_tokens > 0
                else tok
            )
        )
        badge = "1 pass" if passes == 1 else f"{passes} passes"
        color = "#4ade80" if passes == 1 else "#fbbf24"
        y_lbl = max(pad_t + 12, y_per_op - 5)
        parts.append(
            f'<text x="{cx:.1f}" y="{y_lbl:.1f}" text-anchor="middle" '
            f'font-size="10" fill="{color}">{total_mib:.1f}M ({badge})</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t + ph + 18}" text-anchor="middle" '
            f'font-size="11" fill="#cbd5e1">T={tok}</text>'
        )

    parts.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        'font-size="11" fill="#94a3b8">'
        f"Active Tokens per Step (T) - Blue: {sweep.dma_buffers}-Slot Weight Buffers | "
        "Green: [T, H] Residual | Amber: Operator Scratchpad</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


SVG_MIN_FONT_PX = 13
# Width of one glyph as a fraction of font size, a conservative upper bound for
# the system sans-serif stack on digits, Latin letters, and punctuation.
_SVG_GLYPH_EM = 0.62
_SVG_TICKS = (1, 8, 64, 512, 4096, 32768)
# Okabe-Ito colorblind-safe palette; each scope also has its own marker shape.
_SCOPE_MARKS: Mapping[str, tuple[str, str]] = {
    "whole_step": ("#009E73", "circle"),
    "whole_step_repass": ("#D55E00", "triangle"),
    "layer_group": ("#56B4E9", "diamond"),
    "single_layer": ("#E69F00", "diamond"),
    "per_op": ("#CC79A7", "square"),
}


def svg_text_width(text: str, font_px: float = SVG_MIN_FONT_PX) -> float:
    """Return an upper-bound pixel width of `text` at `font_px`."""
    return len(text) * font_px * _SVG_GLYPH_EM


def _svg_text(
    x: float, y: float, text: str, *, anchor: str = "start", fill: str = "#e2e8f0"
) -> str:
    """Return one SVG text element at the report's minimum font size."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{SVG_MIN_FONT_PX}" fill="{fill}">{_esc(text)}</text>'
    )


def _svg_marker(shape: str, cx: float, cy: float, color: str, title: str) -> str:
    """Return a circle, diamond, square, or triangle marker with a tooltip."""
    r = 6.0
    tip = f"<title>{_esc(title)}</title>"
    common = f'fill="{color}" stroke="#0f172a" stroke-width="1.5"'
    if shape == "circle":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" {common}>{tip}</circle>'
    if shape == "square":
        return (
            f'<rect x="{cx - r:.1f}" y="{cy - r:.1f}" width="{2 * r}" '
            f'height="{2 * r}" {common}>{tip}</rect>'
        )
    if shape == "triangle":
        pts = f"{cx:.1f},{cy - r - 1:.1f} {cx + r + 1:.1f},{cy + r:.1f} "
        pts += f"{cx - r - 1:.1f},{cy + r:.1f}"
        return f'<polygon points="{pts}" {common}>{tip}</polygon>'
    pts = f"{cx:.1f},{cy - r - 1:.1f} {cx + r + 1:.1f},{cy:.1f} "
    pts += f"{cx:.1f},{cy + r + 1:.1f} {cx - r - 1:.1f},{cy:.1f}"
    return f'<polygon points="{pts}" {common}>{tip}</polygon>'


def render_regime_dot_plot_svg(sweep: AdvisorSweepReport) -> str:
    """Render one row per workload on a log axis of tokens per core."""
    regimes = sweep.regimes
    right = [f"{e.recommended_scope}, {e.step_bound_us:,.0f} us" for e in regimes]
    pad, gap, row_h, plot_w = 12.0, 14.0, 26.0, 440.0
    label_w = max(svg_text_width(e.label) for e in regimes)
    right_w = max(svg_text_width(t) for t in right)
    x0 = pad + label_w + gap
    x1 = x0 + plot_w
    marks = (
        ("VMEM fit, whole_step", sweep.max_vmem_tokens, "#009E73"),
        ("VMEM fit, single_layer", sweep.single_layer_vmem_tokens, "#E69F00"),
        ("compute-bound from", sweep.ridge_tokens, "#cbd5e1"),
    )
    hi = max(4096, *(e.active_tokens for e in regimes), *(v for _, v, _ in marks))
    ticks = [t for t in _SVG_TICKS if t <= hi]
    hi = max(hi, ticks[-1])
    log_hi = math.log2(hi)

    def sx(tok: float) -> float:
        """Map tokens per core to an x coordinate on the log2 axis."""
        return x0 + math.log2(max(1.0, float(tok))) / log_hi * plot_w

    mark_labels = [f"{name}: {val:,} tok" for name, val, _ in marks]
    width = max(
        x1 + gap + right_w + pad,
        *(
            sx(v) + 4 + svg_text_width(t) + pad
            for (_, v, _), t in zip(marks, mark_labels)
        ),
    )
    title = "Where each workload lands: tokens per core vs on-chip limits"
    width = max(width, pad + svg_text_width(title, 15) + pad)
    width = max(
        width,
        *(
            pad + 22 + svg_text_width(f"{n}: {_SCOPE_HINTS[n]}") + pad
            for n in _SCOPE_MARKS
        ),
    )
    top = 30.0 + 18.0 * len(marks) + 8.0
    bottom = top + row_h * len(regimes)
    legend_y = bottom + 58.0
    height = legend_y + 20.0 * len(_SCOPE_MARKS) + 8.0

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg" role="img">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#0f172a"/>',
        f'<text x="{pad}" y="20" font-size="15" font-weight="600" '
        f'fill="#f8fafc">{_esc(title)}</text>',
    ]
    for i, ((_, val, color), text) in enumerate(zip(marks, mark_labels)):
        x = sx(val)
        y_lbl = 30.0 + 18.0 * (i + 1)
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_lbl + 4:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="5,4"/>'
        )
        parts.append(_svg_text(x + 4, y_lbl, text))
    for i, (entry, rtext) in enumerate(zip(regimes, right)):
        cy = top + row_h * i + row_h / 2
        color, shape = _SCOPE_MARKS.get(entry.recommended_scope, ("#cbd5e1", "circle"))
        parts.append(
            f'<line x1="{x0:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{cy:.1f}" '
            'stroke="#334155" stroke-width="1"/>'
        )
        parts.append(_svg_text(x0 - gap / 2, cy + 4.5, entry.label, anchor="end"))
        tip = (
            f"{entry.label}: {entry.active_tokens} tokens/core, "
            f"AI {entry.step_arithmetic_intensity} FLOP/B, {entry.layout} -> "
            f"{entry.recommended_scope} ({entry.recommended_moe_mode}), "
            f"{entry.step_bound_us:.1f} us"
        )
        parts.append(_svg_marker(shape, sx(entry.active_tokens), cy, color, tip))
        parts.append(_svg_text(x1 + gap, cy + 4.5, rtext))
    parts.append(
        f'<line x1="{x0:.1f}" y1="{bottom:.1f}" x2="{x1:.1f}" y2="{bottom:.1f}" '
        'stroke="#94a3b8" stroke-width="1"/>'
    )
    for t in ticks:
        x = sx(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 5:.1f}" '
            'stroke="#94a3b8" stroke-width="1"/>'
        )
        parts.append(_svg_text(x, bottom + 20, f"{t:,}", anchor="middle"))
    parts.append(
        _svg_text(
            (x0 + x1) / 2,
            bottom + 40,
            "Active tokens per core per step (log scale)",
            anchor="middle",
            fill="#cbd5e1",
        )
    )
    for i, (name, (color, shape)) in enumerate(_SCOPE_MARKS.items()):
        y = legend_y + 20.0 * i
        parts.append(_svg_marker(shape, pad + 8, y - 4.5, color, name))
        parts.append(_svg_text(pad + 22, y, f"{name}: {_SCOPE_HINTS[name]}"))
    parts.append("</svg>")
    return "\n".join(parts)


def render_advisor_html(sweep: AdvisorSweepReport) -> str:
    """Render a self-contained, zero-CDN HTML report with inline SVG regime charts."""
    rec = sweep.parallelism
    vmem_svg = _render_vmem_svg(sweep)
    regime_svg = render_regime_dot_plot_svg(sweep)
    safe_json = json.dumps(sweep.to_dict(), indent=2).replace("</", "<\\/")

    rows_html = []
    for entry in sweep.regimes:
        scope = _esc(entry.recommended_scope)
        moe_mode = entry.recommended_moe_mode
        moe_hint = _esc(_MOE_MODE_SHORT_HINTS.get(moe_mode, ""))
        rows_html.append(
            "<tr>"
            f"<td><strong>{_esc(entry.label)}</strong></td>"
            f"<td>{_esc(entry.layout)}</td>"
            f"<td>{entry.active_tokens}</td>"
            f"<td>{entry.step_arithmetic_intensity:.2f}</td>"
            f"<td>{entry.num_vmem_splits}</td>"
            f'<td><span class="badge badge-{scope}">{scope}</span></td>'
            f"<td><code>{_esc(moe_mode)}</code>"
            f'<div class="kpi-sub">{moe_hint}</div></td>'
            f"<td>{entry.step_bound_us:.1f}</td>"
            f"<td>{_esc(entry.rationale)}</td>"
            "</tr>"
        )

    rules_html = "".join(f"<li>{_esc(r)}</li>" for r in sweep.tailoring_guide)
    mode_src = rec.source or ("auto-derived" if rec.auto_selected else "manual")
    pre = sweep.prefill_parallelism
    prefill_kpi = (
        f'<div class="kpi-sub">Prefill: {_esc(pre.label())} ({_esc(pre.source)})</div>'
        if pre is not None
        else ""
    )
    scope_items = "".join(
        f"<li><strong><code>{_esc(n)}</code></strong>: {_esc(h)}.</li>"
        for n, h in _SCOPE_HINTS.items()
    )
    moe_items = "".join(
        f"<li><strong><code>{_esc(n)}</code></strong>: {_esc(h)}.</li>"
        for n, h in _MOE_MODE_SHORT_HINTS.items()
    )
    residual_k = sweep.vmem_breakdown_bytes["residual_buffer_bytes"] // 1024
    static_str = (
        f"{sweep.vmem_static_mib:.1f} / {sweep.vmem_budget_mib:.1f} MiB "
        f"({sweep.dma_buffers}-slot DMA)"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Megakernel Advisor Report - {_esc(sweep.model)} ({_esc(sweep.chip)})</title>
<style>
  :root {{
    --bg: #0b1120; --card: #1e293b; --border: #334155;
    --text: #f8fafc; --muted: #94a3b8;
  }}
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    margin: 0; padding: 24px; line-height: 1.5;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1, h2 {{ margin-top: 0; }}
  .grid-kpi {{
    display: grid; gap: 14px; margin-bottom: 20px;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }}
  .kpi-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; }}
  .kpi-val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .kpi-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .grid-charts {{
    display: grid; gap: 16px; margin-bottom: 20px;
    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
  }}
  .chart-svg {{ width: 100%; height: auto; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{
    border-bottom: 1px solid var(--border);
    padding: 10px 12px; text-align: left; vertical-align: top;
  }}
  th {{ color: var(--muted); font-weight: 600; background: #0f172a; }}
  .badge {{ padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
  .badge-whole_step {{ background: #065f46; color: #6ee7b7; }}
  .badge-layer_group, .badge-single_layer {{ background: #78350f; color: #fde68a; }}
  .badge-per_op {{ background: #0c4a6e; color: #7dd3fc; }}
</style>
</head>
<body>
<div class="container">
  <h1>TPU Megakernel Fusion-Scope Advisor: {_esc(sweep.model)}</h1>
  <div class="grid-kpi">
    <div class="card">
      <div class="kpi-label">Target Chip &amp; Mesh</div>
      <div class="kpi-val">{_esc(sweep.chip)} &middot; {_esc(rec.label())}</div>
      <div class="kpi-sub">Decode slice: {_esc(rec.topology)} ({_esc(mode_src)})</div>
      {prefill_kpi}
    </div>
    <div class="card">
      <div class="kpi-label">Full-Step VMEM Ceiling</div>
      <div class="kpi-val">{sweep.max_vmem_tokens} tokens</div>
      <div class="kpi-sub">Static: {static_str}</div>
    </div>
    <div class="card">
      <div class="kpi-label">Per-Token VMEM Slope</div>
      <div class="kpi-val">{sweep.per_token_vmem_bytes / 1024:.1f} KiB/tok</div>
      <div class="kpi-sub">[T, H] Residual {residual_k}K + Operator Scratchpad</div>
    </div>
    <div class="card">
      <div class="kpi-label">Hardware Ridge Crossover</div>
      <div class="kpi-val">{sweep.ridge_dense_flop_per_byte:.1f} FLOP/B</div>
      <div class="kpi-sub">T_ridge &asymp; {sweep.ridge_tokens} tokens/step</div>
    </div>
  </div>
  <div class="grid-charts">
    <div class="card">
      <h2>1. VMEM Capacity &amp; Single-Pass Token Ceiling</h2>
      {vmem_svg}
    </div>
    <div class="card">
      <h2>2. Where Each Workload Lands (Tokens per Core vs On-Chip Limits)</h2>
      {regime_svg}
    </div>
  </div>
  <div class="card" style="margin-bottom: 20px;">
    <h2>3. How to Read the Scope and MoE Mode</h2>
    <p>Scope: where the kernel boundary falls.</p>
    <ul>{scope_items}</ul>
    <p>MoE mode: where tokens are grouped by expert and where the data lives.</p>
    <ul>{moe_items}</ul>
  </div>
  <div class="card" style="margin-bottom: 20px;">
    <h2>4. Request Regime Recommendations</h2>
    <table>
      <thead>
        <tr>
          <th>Request Regime</th><th>Layout</th><th>Active Tokens (T)</th>
          <th>AI (FLOP/B)</th><th>Token Passes</th><th>Scope</th><th>MoE Mode</th>
          <th>Step Bound (us)</th><th>Mechanical Rationale</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
  </div>
  <div class="card">
    <h2>5. How to Tailor to Your Request Mix</h2>
    <ul>{rules_html}</ul>
  </div>
</div>
<script id="advisor-data" type="application/json">{safe_json}</script>
</body>
</html>
"""


def write_advisor_html(sweep: AdvisorSweepReport, path: str | Path) -> Path:
    """Write self-contained HTML regime report to disk."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_advisor_html(sweep), encoding="utf-8")
    return dest
