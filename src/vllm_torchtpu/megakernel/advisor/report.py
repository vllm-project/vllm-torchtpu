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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from . import (
        AdvisorReport,
        AdvisorSweepReport,
        ModelProfile,
        RegimeEntry,
        chips,
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
    "layer_group": "one call per group of N layers, tiled like whole_step",
    "single_layer": "one call per layer",
    "per_op": "one call per operator; activations hand off through HBM",
}
_PROVENANCE_TAGS: tuple[str, ...] = ("DERIVED", "ASSUMED", "MEASURED")
_BAND_PAIR_LEN = 2
# Table 5 verdict labels for scopes that fuse more than one layer per call.
_MEGAKERNEL_LABELS: Mapping[str, str] = {
    "whole_step": "WHOLE-STEP MEGAKERNEL",
    "layer_group": "LAYER-GROUP MEGAKERNEL",
}
# Columns of Table 5; the ledger row spans all of them.
_DECISION_TABLE_COLS = 12


def _esc(val: object) -> str:
    """Escape a value for safe HTML rendering."""
    return "" if val is None else html.escape(str(val))


def format_advisor_summary(
    rep: AdvisorReport, *, explain: bool = False, unresolved: bool = False
) -> str:
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
            f"(build={rep.build_megakernel}, verdict={rep.verdict})"
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
    if explain and rep.metrics:
        lines.append("Provenance Breakdown:")
        lines.extend(
            f"  * [{item['tag']}] {key}: {item['value']} {item['unit']} "
            f"(band=[{item['band'][0]}, {item['band'][1]}], source={item['source']})"
            for key, item in rep.metrics.items()
        )
    if unresolved and rep.unresolved_assumptions:
        lines.append("Unresolved Assumptions:")
        lines.extend(
            f"  * {item['name']} (band={item['band']} {item['unit']}): "
            f"{item['how_to_settle']}"
            for item in rep.unresolved_assumptions
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
    sat = sweep.saturation_parallelism
    if sat is not None:
        layout_lines.append(
            f"Decode Saturation Parallelism: {sat.label()} [{sat.topology}] "
            f"({sat.source})"
        )
    handoff = sweep.disagg_handoff
    if handoff is not None:
        lo_ms, hi_ms = handoff.dcn_xfer_bracket_ms
        layout_lines.append(
            f"Disaggregated P/D DCN Handoff (ISL={handoff.isl}): "
            f"{handoff.total_wire_mb:.2f} MB/req "
            f"(KV={handoff.kv_wire_mb:.2f} MB + "
            f"state={handoff.state_wire_mb:.2f} MB) -> "
            f"{handoff.dcn_xfer_ms:.2f} ms [{lo_ms:.2f}..{hi_ms:.2f} ms] "
            f"at {handoff.dcn_bw_gbps:.0f} Gbps/chip [{handoff.dcn_provenance}]"
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
    width, height = 960, 310
    pad_l, pad_r, pad_t, pad_b = 64, 32, 42, 50
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    static_vmem_mib = sweep.vmem_static_mib
    residual_per_tok_mib = sweep.vmem_breakdown_bytes["residual_buffer_bytes"] / (
        1 << 20
    )
    per_op_per_tok_mib = (
        sweep.per_token_vmem_bytes - sweep.vmem_breakdown_bytes["residual_buffer_bytes"]
    ) / (1 << 20)

    token_ticks = [4, 8, 16, 64, 128, 256, 512]
    max_total_mib = max(
        static_vmem_mib + (residual_per_tok_mib + per_op_per_tok_mib) * tok
        for tok in token_ticks
    )
    cap_mib = max(sweep.vmem_budget_mib * 1.15, max_total_mib * 1.08)
    bar_w = pw / (len(token_ticks) * 1.8)
    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
        ),
        (
            f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" '
            'fill="var(--chart-bg, #181A2B)" '
            'stroke="var(--chart-grid, #2D3150)" rx="6"/>'
        ),
    ]
    y_cap = pad_t + ph - min(1.0, sweep.vmem_budget_mib / cap_mib) * ph
    parts.append(
        f'<line x1="{pad_l}" y1="{y_cap:.1f}" x2="{pad_l + pw}" y2="{y_cap:.1f}" '
        'stroke="#EF4444" stroke-width="2.5" stroke-dasharray="8,5"/>'
    )
    lbl_text = (
        f"Usable VMEM Ceiling: {sweep.vmem_budget_mib:.1f} MiB "
        f"(T_max={sweep.max_vmem_tokens} tokens)"
    )
    parts.append(
        f'<rect x="{pad_l + pw - 300}" y="{pad_t - 26}" width="296" height="22" '
        'fill="var(--card-inner, #1C1E32)" stroke="#EF4444" stroke-width="1.5" rx="4"/>'
    )
    parts.append(
        f'<text x="{pad_l + pw - 152}" y="{pad_t - 11}" text-anchor="middle" '
        f'font-size="11" font-weight="700" fill="#FCA5A5">{lbl_text}</text>'
    )

    for idx, tok in enumerate(token_ticks):
        cx = pad_l + (idx + 0.5) * (pw / len(token_ticks))
        x0 = cx - bar_w / 2
        residual_mib = residual_per_tok_mib * tok
        per_op_mib = per_op_per_tok_mib * tok
        total_mib = static_vmem_mib + residual_mib + per_op_mib
        h_static = (static_vmem_mib / cap_mib) * ph
        h_residual = (residual_mib / cap_mib) * ph
        h_per_op = (per_op_mib / cap_mib) * ph

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
        color = "#10B981" if passes == 1 else "#F59E0B"
        y_lbl = max(pad_t + 16, y_per_op - 6)
        parts.append(
            f'<text x="{cx:.1f}" y="{y_lbl:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="600" fill="{color}">'
            f"{total_mib:.1f}M ({badge})</text>"
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t + ph + 20}" text-anchor="middle" '
            'font-size="12" font-weight="600" fill="var(--text, #E4E4F7)">'
            f"T={tok}</text>"
        )

    parts.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{height - 10}" text-anchor="middle" '
        'font-size="12" fill="var(--muted, #9D9EC4)">'
        f"Active Tokens per Step (T) &middot; Blue: {sweep.dma_buffers}-Slot "
        "Weight Buffers | "
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
    "layer_group": ("#56B4E9", "diamond"),
    "single_layer": ("#E69F00", "diamond"),
    "per_op": ("#CC79A7", "square"),
}


def svg_text_width(text: str, font_px: float = SVG_MIN_FONT_PX) -> float:
    """Return an upper-bound pixel width of `text` at `font_px`."""
    return len(text) * font_px * _SVG_GLYPH_EM


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "start",
    fill: str = "var(--text, #E4E4F7)",
) -> str:
    """Return one SVG text element at the report's minimum font size."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{SVG_MIN_FONT_PX}" fill="{fill}">{_esc(text)}</text>'
    )


def _svg_marker(shape: str, cx: float, cy: float, color: str, title: str) -> str:
    """Return a circle, diamond, square, or triangle marker with a tooltip."""
    r = 7.0
    tip = f"<title>{_esc(title)}</title>"
    common = f'fill="{color}" stroke="var(--chart-bg, #181A2B)" stroke-width="2.0"'
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
    pad, gap, row_h, plot_w = 16.0, 16.0, 28.0, 520.0
    label_w = max((svg_text_width(e.label) for e in regimes), default=0.0)
    right_w = max((svg_text_width(t) for t in right), default=0.0)
    x0 = pad + label_w + gap
    x1 = x0 + plot_w
    marks = (
        ("whole_step VMEM ceiling", "#009E73"),
        ("single_layer VMEM ceiling", "#E69F00"),
        ("compute-bound point", "#38BDF8"),
    )
    row_limits = [_row_limits(e) for e in regimes]
    hi = max(
        (
            4096,
            *(e.active_tokens for e in regimes),
            *(v for lim in row_limits for v in lim if v > 0),
        )
    )
    ticks = [t for t in _SVG_TICKS if t <= hi]
    hi = max(hi, ticks[-1])
    log_hi = math.log2(hi)

    def sx(tok: float) -> float:
        """Map tokens per core to an x coordinate on the log2 axis."""
        return x0 + math.log2(max(1.0, float(tok))) / log_hi * plot_w

    mark_labels = [f"| {name} (this row's layout)" for name, _ in marks]
    width = max(
        x1 + gap + right_w + pad,
        *(x0 + 18 + svg_text_width(t) + pad for t in mark_labels),
    )
    title = "Where each workload lands: tokens per core vs on-chip limits"
    width = max(width, pad + svg_text_width(title, 15) + pad)
    width = max(
        width,
        *(
            pad + 24 + svg_text_width(f"{n}: {_SCOPE_HINTS[n]}") + pad
            for n in _SCOPE_MARKS
        ),
    )
    top = 34.0 + 20.0 * len(marks) + 8.0
    bottom = top + row_h * len(regimes)
    legend_y = bottom + 60.0
    height = legend_y + 22.0 * len(_SCOPE_MARKS) + 12.0

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg" role="img">',
        (
            f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
            'fill="var(--chart-bg, #181A2B)" stroke="var(--border, #373B5E)" '
            'stroke-width="1" rx="6"/>'
        ),
        f'<text x="{pad}" y="24" font-size="15" font-weight="700" '
        f'fill="var(--text, #E4E4F7)">{_esc(title)}</text>',
    ]
    for i, ((_, color), text) in enumerate(zip(marks, mark_labels, strict=True)):
        y_lbl = 34.0 + 20.0 * (i + 1)
        parts.append(
            f'<line x1="{x0 + 4:.1f}" y1="{y_lbl - 11:.1f}" x2="{x0 + 4:.1f}" '
            f'y2="{y_lbl + 3:.1f}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(_svg_text(x0 + 14, y_lbl, text[2:], fill=color))
    for i, (entry, rtext) in enumerate(zip(regimes, right, strict=True)):
        cy = top + row_h * i + row_h / 2
        for (name, color), val in zip(marks, row_limits[i], strict=True):
            if val > 0:
                xv = sx(val)
                parts.append(
                    f'<line x1="{xv:.1f}" y1="{cy - 9:.1f}" x2="{xv:.1f}" '
                    f'y2="{cy + 9:.1f}" stroke="{color}" stroke-width="3">'
                    f"<title>{_esc(entry.label)}: {name} = {val:,} tokens/core"
                    "</title></line>"
                )
        color, shape = _SCOPE_MARKS.get(entry.recommended_scope, ("#cbd5e1", "circle"))
        parts.append(
            f'<line x1="{x0:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{cy:.1f}" '
            'stroke="var(--chart-grid, #2D3150)" stroke-width="1.5" '
            'stroke-dasharray="2,4"/>'
        )
        parts.append(
            _svg_text(
                x0 - gap / 2,
                cy + 4.5,
                entry.label,
                anchor="end",
                fill="var(--text, #E4E4F7)",
            )
        )
        tip = (
            f"{entry.label}: {entry.active_tokens} tokens/core, "
            f"AI {entry.step_arithmetic_intensity} FLOP/B, {entry.layout} -> "
            f"{entry.recommended_scope} ({entry.recommended_moe_mode}), "
            f"{entry.step_bound_us:.1f} us"
        )
        parts.append(_svg_marker(shape, sx(entry.active_tokens), cy, color, tip))
        parts.append(_svg_text(x1 + gap, cy + 4.5, rtext, fill="var(--muted, #9D9EC4)"))
    parts.append(
        f'<line x1="{x0:.1f}" y1="{bottom:.1f}" x2="{x1:.1f}" y2="{bottom:.1f}" '
        'stroke="var(--muted, #9D9EC4)" stroke-width="1.5"/>'
    )
    for t in ticks:
        x = sx(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 5:.1f}" '
            'stroke="var(--muted, #9D9EC4)" stroke-width="1.5"/>'
        )
        parts.append(
            _svg_text(
                x, bottom + 22, f"{t:,}", anchor="middle", fill="var(--muted, #9D9EC4)"
            )
        )
    parts.append(
        _svg_text(
            (x0 + x1) / 2,
            bottom + 44,
            "Active tokens per core per step (log scale)",
            anchor="middle",
            fill="var(--text, #E4E4F7)",
        )
    )
    for i, (name, (color, shape)) in enumerate(_SCOPE_MARKS.items()):
        y = legend_y + 22.0 * i
        parts.append(_svg_marker(shape, pad + 8, y - 4.5, color, name))
        parts.append(
            _svg_text(
                pad + 24,
                y,
                f"{name}: {_SCOPE_HINTS[name]}",
                fill="var(--text, #E4E4F7)",
            )
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _row_limits(entry: RegimeEntry) -> tuple[int, int, int]:
    """Return this row's whole_step and single_layer VMEM ceilings and ridge."""
    ceil = {
        str(sc["scope"]): int(str(sc["t_vmem_max"]))
        for sc in entry.scopes
        if sc["feasible"]
    }
    ridge = int(float(str(entry.metrics.get("compute_bound_at", {}).get("value", 0))))
    return ceil.get("whole_step", 0), ceil.get("single_layer", 0), ridge


def _scope_floor_us(entry: RegimeEntry, scope: str) -> float:
    """Return the floor of `scope` on this row, or 0.0 when it is infeasible."""
    for sc in entry.scopes:
        if sc["scope"] == scope and sc["feasible"]:
            return float(str(sc["step_bound_us"]))
    return 0.0


def render_step_band_svg(sweep: AdvisorSweepReport) -> str:
    """Render horizontal one-sided upward uncertainty whiskers for each regime."""
    regimes = sweep.regimes
    pad, gap, row_h, plot_w = 16.0, 16.0, 28.0, 520.0
    label_w = max((svg_text_width(e.label) for e in regimes), default=0.0)
    right = [
        f"{e.step_bound_us:,.1f}..{max(e.step_bound_us, e.step_upper_us):,.1f} us"
        for e in regimes
    ]
    right_w = max((svg_text_width(t) for t in right), default=0.0)
    x0 = pad + label_w + gap
    x1 = x0 + plot_w
    min_us = min((e.step_bound_us for e in regimes), default=50.0)
    max_us = max(
        (
            10.0,
            *(max(e.step_bound_us, e.step_upper_us) for e in regimes),
            *(_scope_floor_us(e, "per_op") for e in regimes),
        )
    )
    domain_min = max(1.0, min(50.0, min_us * 0.75))
    domain_max = max(domain_min * 1.25, max_us * 1.15)
    log_min = math.log10(domain_min)
    log_max = math.log10(domain_max)

    def sx_us(val_us: float) -> float:
        """Map latency in microseconds to horizontal log10 coordinate."""
        clamped = max(domain_min, min(domain_max, float(val_us)))
        t = (math.log10(clamped) - log_min) / max(0.01, log_max - log_min)
        return x0 + max(0.0, min(1.0, t)) * plot_w

    title = "One-sided upward step latency bands: hardware floor to upper bound (us)"
    width = max(
        x1 + gap + right_w + pad,
        pad + svg_text_width(title, 15) + pad,
    )
    top = 62.0
    bottom = top + row_h * len(regimes)
    height = bottom + 38.0
    legend = (
        "dot: floor of the recommended scope | bar: up to the upper bound | "
        "red dot: per_op floor at the same inputs"
    )
    width = max(width, pad + svg_text_width(legend) + pad)

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg" role="img">',
        (
            f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" '
            'fill="var(--chart-bg, #181A2B)" stroke="var(--border, #373B5E)" '
            'stroke-width="1" rx="6"/>'
        ),
        f'<text x="{pad}" y="24" font-size="15" font-weight="700" '
        f'fill="var(--text, #E4E4F7)">{_esc(title)}</text>',
        _svg_text(pad, 46, legend, fill="var(--muted, #9D9EC4)"),
    ]
    for gt in (500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0):
        if domain_min <= gt <= domain_max:
            gx = sx_us(gt)
            parts.append(
                f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{bottom:.1f}" '
                'stroke="var(--chart-grid, #2D3150)" stroke-width="1.5" '
                'stroke-dasharray="3,3"/>'
            )
            parts.append(
                _svg_text(
                    gx,
                    bottom + 18,
                    f"{gt / 1000:g}ms",
                    anchor="middle",
                    fill="var(--muted, #9D9EC4)",
                )
            )

    for idx, (entry, rtext) in enumerate(zip(regimes, right, strict=True)):
        cy = top + row_h * idx + row_h / 2
        lo = float(entry.step_bound_us)
        hi = max(lo, float(entry.step_upper_us))
        x_lo = sx_us(lo)
        x_hi = max(x_lo, sx_us(hi))
        color, _ = _SCOPE_MARKS.get(entry.recommended_scope, ("#38bdf8", "circle"))
        parts.append(
            f'<line x1="{x0:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{cy:.1f}" '
            'stroke="var(--chart-grid, #2D3150)" stroke-width="1.5" '
            'stroke-dasharray="2,4"/>'
        )
        parts.append(
            _svg_text(
                x0 - gap / 2,
                cy + 4.5,
                entry.label,
                anchor="end",
                fill="var(--text, #E4E4F7)",
            )
        )
        parts.append(
            f'<line class="uncertainty-whisker" data-floor="{lo:.2f}" '
            f'data-upper="{hi:.2f}" x1="{x_lo:.2f}" y1="{cy:.2f}" '
            f'x2="{x_hi:.2f}" y2="{cy:.2f}" stroke="{color}" stroke-width="3"/>'
        )
        per_op_us = _scope_floor_us(entry, "per_op")
        if per_op_us > 0.0 and entry.recommended_scope != "per_op":
            x_per_op = sx_us(per_op_us)
            tip = f"per_op floor: {per_op_us:,.1f} us (+{per_op_us - lo:,.1f} us)"
            parts.append(
                f'<circle cx="{x_per_op:.2f}" cy="{cy:.2f}" r="4.0" '
                f'fill="#EF4444" opacity="0.9"><title>{tip}</title></circle>'
            )
        parts.append(
            f'<circle cx="{x_lo:.2f}" cy="{cy:.2f}" r="5.0" fill="{color}" '
            'stroke="var(--chart-bg, #181A2B)" stroke-width="1.5"/>'
        )
        parts.append(_svg_text(x1 + gap, cy + 4.5, rtext, fill="var(--muted, #9D9EC4)"))
    parts.append("</svg>")
    return "\n".join(parts)


def _render_ledger_row(entry: RegimeEntry) -> str:
    """Render a collapsed full-width row listing every number behind one regime."""
    counts = dict.fromkeys(_PROVENANCE_TAGS, 0)
    metric_rows: list[str] = []
    for name, metric in entry.metrics.items():
        tag = str(metric["tag"])
        counts[tag] = counts.get(tag, 0) + 1
        band = metric["band"]
        band_str = f"[{band[0]}, {band[1]}]" if len(band) >= _BAND_PAIR_LEN else "[]"
        metric_rows.append(
            f"<tr><td><code>{_esc(name)}</code></td>"
            f'<td><span class="prov-tag prov-tag-{_esc(tag)}">{_esc(tag)}</span></td>'
            f'<td class="tabular">{_esc(metric["value"])}</td>'
            f"<td>{_esc(metric['unit'])}</td>"
            f'<td class="tabular">{_esc(band_str)}</td>'
            f"<td>{_esc(metric['source'])}</td></tr>"
        )
    scope_rows = [
        f"<tr><td><code>{_esc(sc['scope'])}</code></td>"
        f"<td>{'yes' if sc['feasible'] else 'no'}</td>"
        f'<td class="tabular">{float(str(sc["step_bound_us"])):,.1f}</td>'
        f'<td class="tabular">{float(str(sc["step_upper_us"])):,.1f}</td>'
        f'<td class="tabular">{_esc(sc["ops_per_step"])}</td>'
        f'<td class="tabular">{_esc(sc["n_splits"])}</td></tr>'
        for sc in entry.scopes
    ]
    summary = ", ".join(f"{counts[t]} {t.lower()}" for t in _PROVENANCE_TAGS)
    scope_table = (
        '<table class="ledger-table"><thead><tr><th>Scope</th><th>Feasible</th>'
        '<th class="tabular">Floor (us)</th><th class="tabular">Upper (us)</th>'
        '<th class="tabular">Launches</th><th class="tabular">Passes</th>'
        f"</tr></thead><tbody>{''.join(scope_rows)}</tbody></table>"
        if scope_rows
        else ""
    )
    return (
        f'<tr class="ledger-row"><td colspan="{_DECISION_TABLE_COLS}">'
        '<details class="prov-details">'
        f"<summary>Numbers behind this row: {_esc(summary)}</summary>"
        '<table class="ledger-table"><thead><tr><th>Metric</th><th>Tag</th>'
        '<th class="tabular">Value</th><th>Unit</th><th class="tabular">Band</th>'
        f"<th>Source</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>"
        f"{scope_table}</details></td></tr>"
    )


def _render_flags_cell(entry: RegimeEntry) -> str:
    """Render pill badges for non-empty regime flags."""
    if not entry.flags:
        return '<span class="kpi-sub">none</span>'
    return " ".join(f'<span class="flag-pill">{_esc(fl)}</span>' for fl in entry.flags)


@dataclass(frozen=True)
class _FactsCardData:
    """Resolved hardware/model specs and provenance tags for the top facts card."""

    chip_spec: chips.ChipSpec
    model_prof: ModelProfile
    chip_sources: dict[str, str]
    model_sources: dict[str, str]
    cal_status_badge: str
    legend_html: str
    lg_metric: Mapping[str, object]
    coll_metric: Mapping[str, object]
    cov_metric: Mapping[str, object]
    lg_tag: str
    coll_tag: str
    cov_tag: str


def _resolve_facts_card_data(sweep: AdvisorSweepReport) -> _FactsCardData:
    """Resolve hardware spec, model profile, sources, and calibration status."""
    if sweep.chip_spec is None:
        msg = "sweep.chip_spec is required to render the hardware facts card"
        raise ValueError(msg)
    if sweep.model_profile is None:
        msg = "sweep.model_profile is required to render the model facts card"
        raise ValueError(msg)
    chip_spec = sweep.chip_spec
    model_prof = sweep.model_profile

    chip_sources = dict(sweep.chip_sources or chip_spec.sources or {})
    model_sources = dict(sweep.model_sources or {})

    first_entry = sweep.regimes[0] if sweep.regimes else None
    lg_metric = first_entry.metrics.get("launch_gap_us", {}) if first_entry else {}
    coll_metric = (
        first_entry.metrics.get("collective_latency_us", {}) if first_entry else {}
    )
    cov_metric = first_entry.metrics.get("step_coverage", {}) if first_entry else {}

    handoff = sweep.disagg_handoff
    dcn_tag = handoff.dcn_provenance if handoff is not None else "ASSUMED"
    lg_tag = str(lg_metric.get("tag", "ASSUMED"))
    coll_tag = str(coll_metric.get("tag", "ASSUMED"))
    cov_tag = str(cov_metric.get("tag", "ASSUMED"))
    is_calibrated = "MEASURED" in (lg_tag, coll_tag, cov_tag, dcn_tag)
    measured = [
        m for m in (lg_metric, coll_metric, cov_metric) if m.get("tag") == "MEASURED"
    ]
    meas_source = (
        str(measured[0].get("source", ""))
        if measured
        else ("dcn_bw_gbps" if dcn_tag == "MEASURED" else "")
    )

    if is_calibrated:
        cal_status_badge = (
            f'<span class="badge badge-measured">CALIBRATED TRACE</span> '
            f'<span class="cal-source">Source: <code>{_esc(meas_source)}</code></span>'
        )
    else:
        hint_code = "advisor calibrate &lt;trace.xplane.pb&gt;"
        cal_status_badge = (
            '<span class="badge badge-assumed">UNCALIBRATED PRIORS</span> '
            f'<span class="cal-hint">Calibrate via: <code>{hint_code}</code></span>'
        )
    legend_html = (
        '<span class="prov-tag prov-tag-DERIVED">DERIVED</span>'
        '<span class="prov-tag prov-tag-ASSUMED">ASSUMED</span>'
        + (
            '<span class="prov-tag prov-tag-MEASURED">MEASURED</span>'
            if is_calibrated
            else '<span class="kpi-sub">'
            "MEASURED: none yet (run advisor calibrate)</span>"
        )
    )
    return _FactsCardData(
        chip_spec=chip_spec,
        model_prof=model_prof,
        chip_sources=chip_sources,
        model_sources=model_sources,
        cal_status_badge=cal_status_badge,
        legend_html=legend_html,
        lg_metric=lg_metric,
        coll_metric=coll_metric,
        cov_metric=cov_metric,
        lg_tag=lg_tag,
        coll_tag=coll_tag,
        cov_tag=cov_tag,
    )


def _render_decision_table_rows(sweep: AdvisorSweepReport) -> list[str]:
    """Render HTML table rows for Table 5 (Decision Table)."""
    rows_html = []
    for entry in sweep.regimes:
        scope = _esc(entry.recommended_scope)
        moe_mode = entry.recommended_moe_mode
        hi_us = max(entry.step_bound_us, entry.step_upper_us)
        flags_cell = _render_flags_cell(entry)

        tput_val = float(
            entry.metrics.get("throughput_tok_s_per_chip", {}).get("value", 0.0)
        )
        fg_val = float(entry.metrics.get("fusion_gain_max", {}).get("value", 0.0))
        gain_str = f"+{fg_val * 100:.1f}%" if fg_val > 0 else "+0.0%"
        gain_cls = "text-gain" if fg_val > 0 else "text-muted"

        verdict_str = entry.verdict
        if verdict_str.startswith("tie(measure:"):
            knob = verdict_str[len("tie(measure:") : -1].strip()
            verdict_cell = (
                '<span class="verdict-tag verdict-tie">TIE</span>'
                f'<div class="kpi-sub">measure: <code>{_esc(knob)}</code></div>'
            )
        elif verdict_str.startswith("tie(objective_split:"):
            split_info = verdict_str[len("tie(objective_split:") : -1].strip()
            if entry.recommended_scope in _MEGAKERNEL_LABELS:
                label = _MEGAKERNEL_LABELS[entry.recommended_scope]
                tag_cls = "verdict-fusion"
            elif entry.recommended_scope == "single_layer":
                label = "PER-LAYER FUSED KERNEL"
                tag_cls = "verdict-layer"
            else:
                label = "SEPARATE OPS"
                tag_cls = "verdict-separate"
            verdict_cell = (
                f'<span class="verdict-tag {tag_cls}">{label}</span>'
                '<div class="kpi-sub">layout split: '
                f"<code>{_esc(split_info)}</code></div>"
            )
        elif entry.recommended_scope in _MEGAKERNEL_LABELS:
            label = _MEGAKERNEL_LABELS[entry.recommended_scope]
            verdict_cell = f'<span class="verdict-tag verdict-fusion">{label}</span>'
        elif entry.recommended_scope == "single_layer":
            verdict_cell = (
                '<span class="verdict-tag verdict-layer">PER-LAYER FUSED KERNEL</span>'
            )
        else:
            verdict_cell = (
                '<span class="verdict-tag verdict-separate">SEPARATE OPS</span>'
            )

        rows_html.append(
            "<tr>"
            f"<td><strong>{_esc(entry.label)}</strong></td>"
            f"<td><code>{_esc(entry.layout)}</code></td>"
            f'<td class="tabular">{entry.active_tokens:,}</td>'
            f'<td class="tabular">{entry.step_arithmetic_intensity:.2f}</td>'
            f'<td class="tabular">{entry.num_vmem_splits}</td>'
            f'<td><span class="badge badge-{scope}">{scope}</span>'
            f'<div class="kpi-sub"><code>{_esc(moe_mode)}</code></div></td>'
            f'<td class="tabular">{tput_val:,.1f}</td>'
            f'<td class="tabular {gain_cls}">{gain_str}</td>'
            f'<td class="tabular">{entry.step_bound_us:,.1f}'
            f'<div class="kpi-sub tabular" style="text-align: right;">'
            f"upper: {hi_us:,.1f} us</div></td>"
            f"<td>{verdict_cell}</td>"
            f"<td>{flags_cell}</td>"
            f"<td>{_esc(entry.rationale)}</td>"
            "</tr>"
        )
        rows_html.append(_render_ledger_row(entry))
    return rows_html


def _render_facts_ledger_card(sweep: AdvisorSweepReport) -> str:
    """Render the top Sourced Hardware & Model Facts Ledger HTML card."""
    data = _resolve_facts_card_data(sweep)
    chip_spec = data.chip_spec
    model_prof = data.model_prof
    chip_sources = data.chip_sources
    model_sources = data.model_sources
    rec = sweep.parallelism
    chip_bw_gb = chip_spec.hbm_bw_bytes_per_s / 1e9
    fp8_tflops = chip_spec.fp8_flops_per_s / 1e12
    bf16_tflops = chip_spec.bf16_flops_per_s / 1e12
    ici_bw_gb = chip_spec.ici_egress_bw_bytes_per_s / 1e9
    hbm_cap_gb = rec.hbm_capacity_gb

    num_layers = model_prof.num_layers
    hidden_size = model_prof.hidden_size
    num_attn = model_prof.num_attention_heads
    num_kv = model_prof.num_kv_heads
    head_dim = model_prof.head_dim
    num_exp = model_prof.num_experts
    top_k = model_prof.top_k
    moe_desc = (
        f"E={num_exp} routed experts, top-{top_k}, "
        f"intermediate={model_prof.moe_intermediate_size}"
        if num_exp > 0
        else "Dense MLP"
    )

    handoff = sweep.disagg_handoff
    if handoff is not None:
        h_lo, h_hi = handoff.dcn_xfer_bracket_ms
        h_tag = handoff.dcn_provenance
        handoff_html = (
            f'<div class="prior-item">'
            f'<span class="prior-name">Host DCN Hand-Off ({handoff.isl} tok):</span>'
            f'<span class="prior-val"><code>{handoff.dcn_bw_gbps:.0f} Gbps/chip '
            f"({handoff.total_wire_mb:.1f} MB &rarr; {handoff.dcn_xfer_ms:.2f} ms "
            f"[{h_lo:.2f}..{h_hi:.2f} ms])</code></span>"
            f'<span class="prov-tag prov-tag-{h_tag}">{h_tag}</span>'
            f"</div>"
        )
    else:
        handoff_html = ""

    return f"""  <div class="card facts-card">
    <div class="facts-header">
      <div>
        <h2 style="margin: 0; font-size: 16px;">
          Hardware &amp; Model Architecture Specifications
        </h2>
        <div class="kpi-sub">Sourced specs below; values computed from them are
          tagged DERIVED in the KPI cards and in each row's number ledger</div>
      </div>
      <div class="prov-legend">{data.legend_html}</div>
    </div>
    <div class="facts-grid">
      <div class="facts-col">
        <h3>Hardware Specifications (<code>{_esc(sweep.chip)}</code>)</h3>
        <table class="facts-table">
          <tr>
            <td>Usable VMEM / Core</td>
            <td><strong>{sweep.vmem_budget_mib:.1f} MiB</strong></td>
            <td>
              <span class="fact-src">
                {_esc(chip_sources.get("vmem_capacity_bytes", "Hardware Spec"))}
              </span>
            </td>
          </tr>
          <tr>
            <td>HBM Capacity / Core</td>
            <td><strong>{hbm_cap_gb:.1f} GB</strong></td>
            <td>
              <span class="fact-src">
                {_esc(chip_sources.get("hbm_capacity_bytes", "Hardware Spec"))}
              </span>
            </td>
          </tr>
          <tr>
            <td>HBM Bandwidth / Core</td>
            <td><strong>{chip_bw_gb:,.0f} GB/s</strong></td>
            <td>
              <span class="fact-src">
                {_esc(chip_sources.get("hbm_bw_bytes_per_s", "Hardware Spec"))}
              </span>
            </td>
          </tr>
          <tr>
            <td>Peak Compute / Core</td>
            <td><strong>
              {fp8_tflops:,.0f} TFLOP/s (FP8) &middot;
              {bf16_tflops:,.0f} TFLOP/s (BF16)
            </strong></td>
            <td>
              <span class="fact-src">
                {_esc(chip_sources.get("fp8_flops_per_s", "Hardware Spec"))}
              </span>
            </td>
          </tr>
          <tr>
            <td>ICI Interconnect Egress</td>
            <td><strong>{ici_bw_gb:,.0f} GB/s/core</strong></td>
            <td>
              <span class="fact-src">
                {_esc(chip_sources.get("ici_egress_bw_bytes_per_s", "Hardware Spec"))}
              </span>
            </td>
          </tr>
        </table>
      </div>
      <div class="facts-col">
        <h3>Model Architecture Dimensions (<code>{_esc(sweep.model)}</code>)</h3>
        <table class="facts-table">
          <tr>
            <td>Source Configuration</td>
            <td><strong>{_esc(model_prof.model_type)}</strong></td>
            <td>
              <span class="fact-src">
                {_esc(model_sources.get("config", "HF config.json"))}
              </span>
            </td>
          </tr>
          <tr>
            <td>Layers &amp; Hidden Size</td>
            <td><strong>{num_layers} layers &middot; H={hidden_size}</strong></td>
            <td>
              <span class="fact-src">
                {_esc(model_sources.get("num_layers", "HF config.json"))}
              </span>
            </td>
          </tr>
          <tr>
            <td>Attention Geometry</td>
            <td><strong>NH={num_attn}, NKV={num_kv}, HD={head_dim}</strong></td>
            <td>
              <span class="fact-src">
                {_esc(model_sources.get("attention_heads", "HF config.json"))}
              </span>
            </td>
          </tr>
          <tr>
            <td>MoE / MLP Structure</td>
            <td><strong>{_esc(moe_desc)}</strong></td>
            <td>
              <span class="fact-src">
                {_esc(model_sources.get("moe", "Dense MLP"))}
              </span>
            </td>
          </tr>
          <tr>
            <td>Weight &amp; KV Dtypes</td>
            <td>
              <strong>
                weight={model_prof.weight_dtype},
                expert={model_prof.expert_dtype},
                kv={model_prof.kv_dtype}
              </strong>
            </td>
            <td>
              <span class="fact-src">
                {_esc(model_sources.get("dtypes", "HF config.json"))}
              </span>
            </td>
          </tr>
        </table>
      </div>
    </div>
    <div class="priors-bar">
      <div class="priors-header">
        <span class="priors-title">
          Runtime Priors &amp; Empirical Calibration Status
        </span>
        <div>{data.cal_status_badge}</div>
      </div>
      <div class="priors-items">
        <div class="prior-item">
          <span class="prior-name">Launch Gap:</span>
          <span class="prior-val">
            <code>{data.lg_metric.get("value", "1.0..8.0")} us</code>
          </span>
          <span class="prov-tag prov-tag-{data.lg_tag}">{data.lg_tag}</span>
        </div>
        <div class="prior-item">
          <span class="prior-name">Collective Latency:</span>
          <span class="prior-val">
            <code>{data.coll_metric.get("value", "1.0..5.13")} us/hop</code>
          </span>
          <span class="prov-tag prov-tag-{data.coll_tag}">{data.coll_tag}</span>
        </div>
        <div class="prior-item">
          <span class="prior-name">Step Coverage:</span>
          <span class="prior-val">
            <code>{data.cov_metric.get("value", "0.50..0.995")}</code>
          </span>
          <span class="prov-tag prov-tag-{data.cov_tag}">{data.cov_tag}</span>
        </div>
        {handoff_html}
      </div>
    </div>
  </div>"""


def render_advisor_html(sweep: AdvisorSweepReport) -> str:
    """Render a self-contained, zero-CDN HTML report with inline SVG regime charts."""
    rec = sweep.parallelism
    vmem_svg = _render_vmem_svg(sweep)
    regime_svg = render_regime_dot_plot_svg(sweep)
    band_svg = render_step_band_svg(sweep)
    safe_json = json.dumps(sweep.to_dict(), indent=2).replace("</", "<\\/")

    facts_card_html = _render_facts_ledger_card(sweep)
    rows_html = _render_decision_table_rows(sweep)
    rules_html = "".join(f"<li>{_esc(r)}</li>" for r in sweep.tailoring_guide)
    mode_src = rec.source or ("auto-derived" if rec.auto_selected else "manual")
    pre = sweep.prefill_parallelism
    sat = sweep.saturation_parallelism
    prefill_kpi = (
        f'<div class="kpi-sub">Prefill: {_esc(pre.label())} ({_esc(pre.source)})</div>'
        if pre is not None
        else ""
    )
    if sat is not None:
        prefill_kpi += (
            f'<div class="kpi-sub">Saturated Decode: {_esc(sat.label())}</div>'
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
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"/>
<title>Megakernel Advisor Report - {_esc(sweep.model)} ({_esc(sweep.chip)})</title>
<style>
  :root {{
    --bg: #FAF4E5;
    --card: #FFFFFF;
    --card-inner: #F5EFE1;
    --border: #E8DFCF;
    --border-subtle: #F0E8D9;
    --text: #435155;
    --muted: #7E8C8F;
    --accent: #CB4B16;
    --accent-subtle: #FDF2EC;
    --code-bg: #F0E8D9;
    --th-bg: #F5EFE1;
    --chart-bg: #FDFBF7;
    --chart-grid: #E8DFCF;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  }}
  [data-theme="dark"] {{
    --bg: #1F2237;
    --card: #262942;
    --card-inner: #1C1E32;
    --border: #373B5E;
    --border-subtle: #2D3150;
    --text: #E4E4F7;
    --muted: #9D9EC4;
    --accent: #CE3D28;
    --accent-subtle: #3A2326;
    --code-bg: #1C1E32;
    --th-bg: #1F2237;
    --chart-bg: #181A2B;
    --chart-grid: #2D3150;
    --shadow: 0 2px 8px rgba(10, 12, 24, 0.45);
  }}
  body {{
    font-family:
      -apple-system, BlinkMacSystemFont, "Segoe UI",
      Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
    margin: 0; padding: 24px; line-height: 1.5;
    transition: background 0.2s ease, color 0.2s ease;
  }}
  .container {{ max-width: 1240px; margin: 0 auto; }}
  .report-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border);
  }}
  h1 {{ margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.01em; }}
  .report-subtitle {{ font-size: 14px; color: var(--muted); margin-top: 4px; }}
  .theme-btn {{
    background: var(--card); border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600;
    cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
    box-shadow: var(--shadow);
  }}
  .theme-btn:hover {{ border-color: var(--accent); }}
  .grid-kpi {{
    display: grid; gap: 14px; margin-bottom: 20px;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 18px; box-shadow: var(--shadow);
  }}
  .kpi-label {{
    font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.05em; font-weight: 600;
  }}
  .kpi-val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .kpi-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .tabular {{
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }}
  th.tabular, td.tabular {{ text-align: right; }}
  .chart-svg {{ width: 100%; height: auto; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{
    border-bottom: 1px solid var(--border);
    padding: 10px 12px; text-align: left; vertical-align: top;
  }}
  th {{
    color: var(--muted); font-weight: 600;
    background: var(--th-bg); font-size: 12px;
  }}
  .badge {{
    display: inline-block; padding: 2px 8px;
    border-radius: 4px; font-weight: 600; font-size: 11px;
  }}
  .badge-whole_step {{ background: #d1fae5; color: #065f46; }}
  .badge-layer_group, .badge-single_layer {{ background: #fef3c7; color: #92400e; }}
  .badge-per_op {{ background: #e0f2fe; color: #0369a1; }}
  .badge-measured {{ background: #ede9fe; color: #5b21b6; }}
  .badge-assumed {{ background: #ffedd5; color: #9a3412; }}
  [data-theme="dark"] .badge-whole_step {{ background: #064e3b; color: #6ee7b7; }}
  [data-theme="dark"] .badge-layer_group,
  [data-theme="dark"] .badge-single_layer {{
    background: #78350f; color: #fde68a;
  }}
  [data-theme="dark"] .badge-per_op {{ background: #0c4a6e; color: #7dd3fc; }}
  [data-theme="dark"] .badge-measured {{ background: #4c1d95; color: #ddd6fe; }}
  [data-theme="dark"] .badge-assumed {{ background: #7c2d12; color: #fed7aa; }}
  .verdict-tag {{
    display: inline-block; padding: 2px 8px;
    border-radius: 4px; font-weight: 700; font-size: 11px;
  }}
  .verdict-tie {{ background: #fef08a; color: #854d0e; }}
  .verdict-fusion {{ background: #bbf7d0; color: #166534; }}
  .verdict-separate {{ background: #e2e8f0; color: #334155; }}
  .verdict-layer {{ background: #ffedd5; color: #9a3412; }}
  [data-theme="dark"] .verdict-layer {{ background: #7c2d12; color: #fed7aa; }}
  [data-theme="dark"] .verdict-tie {{ background: #713f12; color: #fef08a; }}
  [data-theme="dark"] .verdict-fusion {{ background: #14532d; color: #86efac; }}
  [data-theme="dark"] .verdict-separate {{ background: #334155; color: #cbd5e1; }}
  .prov-tag {{
    display: inline-block; padding: 1px 6px; margin: 1px;
    border-radius: 4px; font-size: 11px; font-weight: 600;
  }}
  .prov-tag-DERIVED {{ background: #dcfce7; color: #166534; }}
  .prov-tag-ASSUMED {{ background: #fef3c7; color: #92400e; }}
  .prov-tag-MEASURED {{ background: #ede9fe; color: #5b21b6; }}
  [data-theme="dark"] .prov-tag-DERIVED {{ background: #065f46; color: #a7f3d0; }}
  [data-theme="dark"] .prov-tag-ASSUMED {{ background: #78350f; color: #fde68a; }}
  [data-theme="dark"] .prov-tag-MEASURED {{ background: #4c1d95; color: #ddd6fe; }}
  .prov-details {{ margin-top: 6px; font-size: 12px; }}
  .prov-details summary {{ cursor: pointer; color: var(--muted); }}
  .ledger-row td {{ background: var(--card-inner); padding-top: 4px; }}
  .ledger-table {{ width: 100%; margin: 8px 0; font-size: 12px; }}
  .text-gain {{ color: #16a34a; font-weight: 600; }}
  [data-theme="dark"] .text-gain {{ color: #4ade80; }}
  .flag-pill {{
    display: inline-block; padding: 2px 8px; margin: 1px;
    border-radius: 999px; font-size: 11px; font-weight: 600;
    background: #fee2e2; color: #991b1b;
  }}
  [data-theme="dark"] .flag-pill {{ background: #7f1d1d; color: #fecaca; }}
  .facts-card {{ margin-bottom: 20px; }}
  .facts-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: 12px; border-bottom: 1px solid var(--border);
  }}
  .facts-grid {{
    display: grid; gap: 18px; margin-top: 14px;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  }}
  .facts-col h3 {{
    margin: 0 0 8px 0; font-size: 13px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .facts-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .facts-table td {{
    padding: 6px 8px; border-bottom: 1px solid var(--border-subtle);
    vertical-align: middle;
  }}
  .facts-table tr:last-child td {{ border-bottom: none; }}
  .fact-src {{ font-size: 11px; color: var(--muted); }}
  .priors-bar {{
    margin-top: 14px; padding: 12px; border-radius: 6px;
    background: var(--card-inner); border: 1px solid var(--border-subtle);
  }}
  .priors-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px; font-size: 12px; font-weight: 600;
  }}
  .priors-title {{
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .priors-items {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; }}
  .prior-item {{ display: inline-flex; align-items: center; gap: 6px; }}
  .prior-name {{ color: var(--muted); }}
  .prior-val {{ font-family: ui-monospace, SFMono-Regular, monospace; }}
  .cal-source {{ font-size: 11px; color: var(--muted); margin-left: 6px; }}
  .cal-hint {{ font-size: 11px; color: var(--muted); margin-left: 6px; }}
  .prov-legend {{ display: inline-flex; gap: 6px; align-items: center; }}
</style>
</head>
<body>
<div class="container">
  <div class="report-header">
    <div>
      <h1>Megakernel Architecture &amp; Regime Advisor</h1>
      <div class="report-subtitle">
        Model: <strong>{_esc(sweep.model)}</strong> &middot;
        Chip: <strong>{_esc(sweep.chip)}</strong> &middot;
        Parallelism: <strong>{_esc(rec.label())}</strong> ({_esc(mode_src)})
      </div>
    </div>
    <button class="theme-btn" id="theme-toggle" onclick="toggleTheme()"
      type="button" aria-label="Toggle theme">
      <span>&#9680;</span> Theme
    </button>
  </div>

  {facts_card_html}

    <div class="grid-kpi">
    <div class="card">
      <div class="kpi-label">Target Chip &amp; Mesh</div>
      <div class="kpi-val">{_esc(sweep.chip)} &middot; {_esc(rec.label())}</div>
      <div class="kpi-sub">Decode slice: {_esc(rec.topology)} ({_esc(mode_src)})</div>
      {prefill_kpi}
    </div>
    <div class="card">
      <div class="kpi-label">Full-Step VMEM Ceiling
        <span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>
      <div class="kpi-val tabular">{sweep.max_vmem_tokens:,} tokens/core</div>
      <div class="kpi-sub">Static: {static_str}; decode layout {_esc(rec.label())}</div>
    </div>
    <div class="card">
      <div class="kpi-label">Per-Token VMEM Slope
        <span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>
      <div class="kpi-val tabular">{sweep.per_token_vmem_bytes / 1024:.1f} KiB/tok</div>
      <div class="kpi-sub">[T, H] Residual {residual_k}K + Operator Scratchpad</div>
    </div>
    <div class="card">
      <div class="kpi-label">Compute-Bound Point
        <span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>
      <div class="kpi-val tabular">{sweep.ridge_dense_flop_per_byte:.1f} FLOP/B</div>
      <div class="kpi-sub">&asymp; {sweep.ridge_tokens} tokens/core/step on the decode
        layout; each row's own point is in Chart 2 and its ledger</div>
    </div>
  </div>

<div class="card" style="margin-bottom: 20px;">
    <h2 style="font-size: 16px;">1. VMEM Capacity &amp; Single-Pass Token Ceiling
      (decode layout {_esc(rec.label())})</h2>
    {vmem_svg}
  </div>
  <div class="card" style="margin-bottom: 20px;">
    <h2 style="font-size: 16px;">
      2. Where Each Workload Lands (Tokens per Core vs On-Chip Limits)
    </h2>
    {regime_svg}
  </div>
  <div class="card" style="margin-bottom: 20px;">
    <h2 style="font-size: 16px;">3. Step Latency Floor vs Upper Uncertainty Band</h2>
    {band_svg}
  </div>
  <div class="card" style="margin-bottom: 20px;">
    <h2 style="font-size: 16px;">4. How to Read the Scope and MoE Mode</h2>
    <p>Scope: where the kernel boundary falls.</p>
    <ul>{scope_items}</ul>
    <p>MoE mode: where tokens are grouped by expert and where the data lives.</p>
    <ul>{moe_items}</ul>
  </div>
  <div class="card" style="margin-bottom: 20px;">
    <h2 style="font-size: 16px;">5. Request Regime Recommendations</h2>
    <p class="kpi-sub">Step bounds are hardware floors, not predictions: on the
      silicon validation corpus they range from well below to near measured step
      time. A small margin to the runner-up is a close call; A/B the two scopes
      before building.</p>
    <table>
      <thead>
        <tr>
          <th>Request Regime</th>
          <th>Layout</th>
          <th class="tabular">Tokens (T)</th>
          <th class="tabular">AI (FLOP/B)</th>
          <th class="tabular">Passes</th>
          <th>Scope &amp; MoE</th>
          <th class="tabular">Throughput (tok/s/chip)</th>
          <th class="tabular">Gain vs per_op</th>
          <th class="tabular">Step Bound (us)</th><th>Verdict &amp; Action</th>
          <th>Flags</th>
          <th>Mechanical Rationale</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
  </div>
  <div class="card">
    <h2 style="font-size: 16px;">6. How to Tailor to Your Request Mix
      (thresholds from the decode layout {_esc(rec.label())})</h2>
    <ul>{rules_html}</ul>
  </div>
</div>
<script id="advisor-data" type="application/json">{safe_json}</script>
<script>
function toggleTheme() {{
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const target = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', target);
  try {{ localStorage.setItem('mk_advisor_theme', target); }} catch(e) {{}}
}}
(function() {{
  try {{
    const saved = localStorage.getItem('mk_advisor_theme');
    if (saved) {{
      document.documentElement.setAttribute('data-theme', saved);
    }} else {{
      document.documentElement.setAttribute('data-theme', 'dark');
    }}
  }} catch(e) {{}}
}})();
</script>
</body>
</html>
"""


def write_advisor_html(sweep: AdvisorSweepReport, path: str | Path) -> Path:
    """Write self-contained HTML regime report to disk."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_advisor_html(sweep), encoding="utf-8")
    return dest
