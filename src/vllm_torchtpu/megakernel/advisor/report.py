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
    "token_masked_vmem": "Stream active experts to VMEM; mask [T, H] rows (T_ep < 32)",
    "expert_grouped_vmem": "Compact tokens by expert in VMEM (32 <= T_ep <= T_max)",
    "standalone_ragged_dot": "HBM SparseCore sort + standalone Pallas ragged_dot",
    "dense_mlp": "Dense SwiGLU FFN (no routed experts)",
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
            f"[mode={rep.recommended_fusion_mode}, "
            f"moe={rep.recommended_moe_mode}] "
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
        f"  - {sc.scope:12s} ({sc.fusion_mode:14s}) {sc.step_bound_us:9.1f} us "
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
    mode_tag = "auto-recommended" if rec.auto_selected else "user-specified"
    hdr = (
        f"{'Request Regime':33s} | {'Tokens':>6s} | {'AI (FLOP/B)':>11s} | "
        f"{'Passes':>6s} | {'Fusion Mode':14s} | {'MoE Mode':21s} | {'Step (us)':>10s}"
    )
    lines = [
        f"=== Megakernel Advisor Regime Report: {sweep.model} on {sweep.chip} ===",
        (
            f"Parallelism: TP={rec.tp} x EP={rec.ep} x DP={rec.dp} x PP={rec.pp} "
            f"x DCP={rec.dcp} x PCP={rec.pcp} [{rec.topology}] ({mode_tag})"
        ),
        (
            f"HBM Footprint: {rec.hbm_used_gb:.2f} / {rec.hbm_capacity_gb:.1f} GB/core "
            f"(active weights/step: {rec.w_step_gb:.2f} GB/core)"
        ),
        (
            f"VMEM Ceiling (full_step, {sweep.dma_buffers}-slot DMA): "
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
        "-" * 119,
    ]
    lines.extend(
        f"{entry.label:33s} | {entry.active_tokens:6d} | "
        f"{entry.step_arithmetic_intensity:11.2f} | {entry.num_vmem_splits:6d} | "
        f"{entry.recommended_fusion_mode:14s} | {entry.recommended_moe_mode:21s} | "
        f"{entry.step_bound_us:10.1f}"
        for entry in sweep.regimes
    )
    lines.extend(
        [
            "",
            "Execution Strategy Legend:",
            "  * token_masked_vmem     : Stream active experts to VMEM; "
            "apply [T, H] routing mask (avoids sort when T_per_ep < 32).",
            "  * expert_grouped_vmem   : Compact tokens by expert ID inside VMEM "
            "into 32-row tiles (32 <= T_per_ep <= T_max).",
            "  * standalone_ragged_dot : Out-of-kernel SparseCore HBM sort + "
            "standalone Pallas ragged_dot (modular_kernel path).",
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
    sublayer_per_tok_mib = (
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
        sublayer_mib = sublayer_per_tok_mib * tok
        total_mib = static_vmem_mib + residual_mib + sublayer_mib
        h_static = min(ph, (static_vmem_mib / cap_mib) * ph)
        h_residual = min(ph - h_static, (residual_mib / cap_mib) * ph)
        h_sublayer = min(ph - h_static - h_residual, (sublayer_mib / cap_mib) * ph)

        y_static = pad_t + ph - h_static
        y_residual = y_static - h_residual
        y_sublayer = y_residual - h_sublayer
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
            f'<rect x="{x0:.1f}" y="{y_sublayer:.1f}" width="{bar_w:.1f}" '
            f'height="{h_sublayer:.1f}" fill="#f59e0b">'
            f"<title>T={tok}: Shared Operator Scratchpad = {sublayer_mib:.2f} MiB "
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
        y_lbl = max(pad_t + 12, y_sublayer - 5)
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


def _render_regime_map_svg(sweep: AdvisorSweepReport) -> str:
    """Render inline SVG phase map of Arithmetic Intensity vs Active Tokens T."""
    width, height = 680, 260
    pad_l, pad_r, pad_t, pad_b = 64, 28, 28, 44
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    max_t = max(
        [4096, sweep.max_vmem_tokens * 2, *(r.active_tokens for r in sweep.regimes)]
    )
    max_ai = max(
        [
            sweep.ridge_dense_flop_per_byte * 2.2,
            *(r.step_arithmetic_intensity for r in sweep.regimes),
        ]
    )

    def sx(tok: float) -> float:
        """Map active token count to horizontal SVG coordinate."""
        l_min, l_max = 0.0, math.log2(max(2.0, float(max_t)))
        l_val = math.log2(max(1.0, float(tok)))
        return pad_l + (l_val - l_min) / (l_max - l_min) * pw

    def sy(ai: float) -> float:
        """Map arithmetic intensity to vertical SVG coordinate."""
        l_min, l_max = 0.0, math.log10(max(10.0, float(max_ai) + 1.0))
        l_val = math.log10(max(1.0, float(ai) + 1.0))
        return pad_t + ph - (l_val - l_min) / (l_max - l_min) * ph

    x_tmax = sx(max(1, sweep.max_vmem_tokens))
    y_ridge = sy(sweep.ridge_dense_flop_per_byte)
    zone_w = max(0.0, x_tmax - pad_l)
    zone_h = max(0.0, pad_t + ph - y_ridge)

    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
        ),
        (
            f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" '
            'fill="#0f172a" stroke="#334155"/>'
        ),
        (
            f'<rect x="{pad_l}" y="{y_ridge:.1f}" width="{zone_w:.1f}" '
            f'height="{zone_h:.1f}" fill="#064e3b" fill-opacity="0.35"/>'
        ),
        f'<line x1="{x_tmax:.1f}" y1="{pad_t}" x2="{x_tmax:.1f}" y2="{pad_t + ph}" '
        'stroke="#10b981" stroke-width="1.8" stroke-dasharray="4,4"/>',
        f'<line x1="{pad_l}" y1="{y_ridge:.1f}" x2="{pad_l + pw}" y2="{y_ridge:.1f}" '
        'stroke="#f43f5e" stroke-width="1.8" stroke-dasharray="4,4"/>',
        f'<text x="{x_tmax + 5:.1f}" y="{pad_t + 14}" font-size="10" fill="#6ee7b7">'
        f"T_max={sweep.max_vmem_tokens}</text>",
        f'<text x="{pad_l + 6}" y="{y_ridge - 6:.1f}" font-size="10" fill="#fda4af">'
        f"Ridge={sweep.ridge_dense_flop_per_byte:.0f} FLOP/B</text>",
    ]

    mode_colors = {
        "full_step": "#22c55e",
        "segment": "#f59e0b",
        "layer": "#eab308",
        "modular_kernel": "#38bdf8",
    }
    for entry in sweep.regimes:
        px = sx(entry.active_tokens)
        py = sy(entry.step_arithmetic_intensity)
        col = mode_colors.get(entry.recommended_fusion_mode, "#94a3b8")
        fusion_mode = entry.recommended_fusion_mode
        moe_mode = entry.recommended_moe_mode
        ai_val = entry.step_arithmetic_intensity
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" '
            f'fill="{col}" stroke="#0f172a" stroke-width="1.5">'
            f"<title>{_esc(entry.label)}: T={entry.active_tokens}, "
            f"AI={ai_val} FLOP/B -> {fusion_mode} ({moe_mode})</title></circle>"
        )
        tx = min(pad_l + pw - 55, px + 9)
        ty = max(pad_t + 12, py - 6)
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="10" fill="#e2e8f0">'
            f"T={entry.active_tokens} ({fusion_mode})</text>"
        )

    parts.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        'font-size="11" fill="#94a3b8">'
        "Log2 Active Tokens (T) vs Log10 Arithmetic Intensity (FLOP/B) - "
        "Green Zone: Full-Step Megakernel</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_advisor_html(sweep: AdvisorSweepReport) -> str:
    """Render a self-contained, zero-CDN HTML report with inline SVG regime charts."""
    rec = sweep.parallelism
    vmem_svg = _render_vmem_svg(sweep)
    regime_svg = _render_regime_map_svg(sweep)
    safe_json = json.dumps(sweep.to_dict(), indent=2).replace("</", "<\\/")

    rows_html = []
    for entry in sweep.regimes:
        fusion_mode = _esc(entry.recommended_fusion_mode)
        moe_mode = entry.recommended_moe_mode
        moe_hint = _esc(_MOE_MODE_SHORT_HINTS.get(moe_mode, ""))
        rows_html.append(
            "<tr>"
            f"<td><strong>{_esc(entry.label)}</strong></td>"
            f"<td>{entry.active_tokens}</td>"
            f"<td>{entry.step_arithmetic_intensity:.2f}</td>"
            f"<td>{entry.num_vmem_splits}</td>"
            f'<td><span class="badge badge-{fusion_mode}">{fusion_mode}</span> '
            f"<small>({_esc(entry.recommended_scope)})</small></td>"
            f"<td><code>{_esc(moe_mode)}</code>"
            f'<div class="kpi-sub">{moe_hint}</div></td>'
            f"<td>{entry.step_bound_us:.1f}</td>"
            f"<td>{_esc(entry.rationale)}</td>"
            "</tr>"
        )

    rules_html = "".join(f"<li>{_esc(r)}</li>" for r in sweep.tailoring_guide)
    mode_src = "auto-derived" if rec.auto_selected else "manual"
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
  .badge-full_step {{ background: #065f46; color: #6ee7b7; }}
  .badge-segment, .badge-layer {{ background: #78350f; color: #fde68a; }}
  .badge-modular_kernel {{ background: #0c4a6e; color: #7dd3fc; }}
</style>
</head>
<body>
<div class="container">
  <h1>TPU Megakernel Fusion-Scope Advisor: {_esc(sweep.model)}</h1>
  <div class="grid-kpi">
    <div class="card">
      <div class="kpi-label">Target Chip &amp; Mesh</div>
      <div class="kpi-val">{_esc(sweep.chip)} &middot; TP{rec.tp}&times;EP{rec.ep}</div>
      <div class="kpi-sub">Slice: {_esc(rec.topology)} ({mode_src})</div>
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
      <h2>2. Request Regime Phase Map (VMEM vs Compute Ridge)</h2>
      {regime_svg}
    </div>
  </div>
  <div class="card" style="margin-bottom: 20px;">
    <h2>3. Execution Strategy Guide (Fusion Mode &amp; MoE Mode)</h2>
    <ul>
      <li><strong><code>full_step</code> (cross_layer)</strong>: Persistent
        single-launch megakernel across all layers; keeps <code>[T, H]</code>
        residual activations in VMEM and overlaps cross-layer weight DMAs with
        in-kernel remote-VMEM collectives.</li>
      <li><strong><code>layer</code> (single_layer) / <code>segment</code>
        (stage)</strong>: Single-layer or pipeline-segment fused kernel; retains
        <code>[T, H]</code> in VMEM within a layer when cross-layer
        multi-buffering exceeds VMEM.</li>
      <li><strong><code>modular_kernel</code> (sublayer)</strong>: Standard
        per-operator Pallas kernels (<code>gmm_v2</code>,
        <code>ragged_paged_attention</code>, <code>gdn</code>) with HBM
        activation handoffs and XLA collectives.</li>
      <li><strong><code>token_masked_vmem</code> (T_per_ep &lt; 32)</strong>:
        Streams only routed experts into VMEM and contracts each against
        <code>[T, H]</code> with a boolean token mask (<code>jnp.where</code>),
        avoiding dynamic token sorting below the TPU FP8 <code>32</code>-row MXU
        floor.</li>
      <li><strong><code>expert_grouped_vmem</code> (32 &le; T_per_ep &le;
        T_max)</strong>: Compacts tokens by expert ID directly inside VMEM into
        <code>32</code>-row aligned tiles before matmul, eliminating HBM sort
        round-trips.</li>
      <li><strong><code>standalone_ragged_dot</code>
        (modular_kernel)</strong>: Globally sorts tokens in HBM via SparseCore
        gather-reduce and executes standalone Pallas <code>gmm_v2</code> /
        <code>ragged_dot</code>.</li>
    </ul>
  </div>
  <div class="card" style="margin-bottom: 20px;">
    <h2>4. Request Regime Recommendations (12-Regime Matrix)</h2>
    <table>
      <thead>
        <tr>
          <th>Request Regime</th><th>Active Tokens (T)</th><th>AI (FLOP/B)</th>
          <th>Token Passes</th><th>Recommended Fusion</th><th>MoE Mode</th>
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
