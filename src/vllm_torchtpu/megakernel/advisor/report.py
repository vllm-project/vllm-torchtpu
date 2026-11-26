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
"""Self-contained static HTML/SVG audit report and offline bucket-schedule export."""

from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .assets import load_report_css
from .constants import (
    BYTES_PER_MIB,
    DECISION_TABLE_COLS,
    HBM_UTILIZATION_CAP,
    MEGAKERNEL_LABELS,
    MIN_PROVENANCE_BAND_LEN,
    MOE_MODE_SHORT_HINTS,
    PHASE_SORT_ORDER,
    PROVENANCE_TAGS,
    SCOPE_HINTS,
    SCOPE_MARKS,
    SVG_GLYPH_EM,
    SVG_MIN_FONT_PX,
    SVG_TICKS,
)
from .models import has_dsa_profile
from .roofline import evaluate_scopes
from .specs import (
    BUCKET_CONSUMER_STATUS,
    BUCKET_SCHEDULE_SCHEMA,
    AdvisorReport,
    AdvisorSweepReport,
    BucketScheduleEntry,
    BucketScheduleReport,
    MoEMode,
    WorkloadSpec,
    as_scalar_number,
)
from .text_format import (
    format_ring_slots,
    gain_basis_label,
    recommended_scope_dict,
    vector_register_load_cap_line,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from . import chips
    from .models import ModelProfile
    from .specs import LayerGroupSearchReport, ProvenanceMetric, RegimeEntry

__all__ = [
    "build_bucket_schedule",
    "render_advisor_html",
    "render_regime_dot_plot_svg",
    "render_step_band_svg",
    "svg_text_width",
    "write_advisor_html",
    "write_bucket_schedule",
]

MS_PER_US = 1000.0
HEADROOM_EPS_MIB = 1e-6
UNKNOWN_PHASE_SORT_ORDER = 99
MIN_PLOT_TOKENS = 4096
VMEM_TOKEN_TICKS: tuple[int, ...] = (4, 8, 16, 64, 128, 256, 512)
DECADE_MULTS: tuple[float, ...] = (1.0, 2.0, 5.0)
SVG_PAD_PX = 16.0
SVG_GAP_PX = 16.0
SVG_TITLE_FONT_PX = 15
SVG_BADGE_FONT_PX = 11
MOE_MODE_MAP: dict[object, MoEMode] = {
    "masked_on_chip": "masked_on_chip",
    "grouped_on_chip": "grouped_on_chip",
    "grouped_in_hbm": "grouped_in_hbm",
    "dense_mlp": "dense_mlp",
}
MEGA_ONLY_FLAGS: frozenset[str] = frozenset(
    {"multi_call_unmodeled", "microbatch_rerun", "above_default_buckets"}
)
HIDDEN_TABLE_FLAGS: frozenset[str] = frozenset({"core_limits_pinned"})
SCOPE_LABELS_HTML: Mapping[str, str] = {
    **MEGAKERNEL_LABELS,
    "single_layer": "PER-LAYER FUSED KERNEL",
    "per_op": "SEPARATE OPS",
}
DOT_PLOT_MARKS: tuple[tuple[str, str], ...] = (
    ("whole_step VMEM ceiling", "#009E73"),
    ("single_layer VMEM ceiling", "#E69F00"),
    ("compute-bound point", "#38BDF8"),
)
ROLE_DISPLAY_ORDER: tuple[str, ...] = ("decode", "prefill", "saturation")
CALIBRATED_PRIOR_KEYS: tuple[str, ...] = (
    "launch_gap_us",
    "collective_latency_us",
    "collective_call_us_xla",
    "collective_call_us_in_kernel",
    "step_coverage",
    "per_op_step_us",
    "host_head_us",
    "calibrated_host_head_us",
)
CHART_GRID = "var(--chart-grid, #2D3150)"
CHART_BG = "var(--chart-bg, #181A2B)"
MUTED = "var(--muted, #9D9EC4)"
TEXT_COLOR = "var(--text, #E4E4F7)"
CALIBRATE_CMD = "python -m vllm_torchtpu.megakernel.advisor calibrate"


def _escape_html(value: object) -> str:
    """Escape an arbitrary value for HTML and SVG attributes or text."""
    return "" if value is None else html.escape(str(value), quote=True)


def svg_text_width(text: str, font_px: float = SVG_MIN_FONT_PX) -> float:
    """Return an upper-bound pixel width estimate for `text` at `font_px`."""
    return len(text) * font_px * SVG_GLYPH_EM


def _scalar_float(raw: object, default: float = 0.0) -> float:
    """Extract a finite float from `raw`, falling back to `default`."""
    parsed = as_scalar_number(raw)
    return default if parsed is None else parsed


def _scalar_int(raw: object, default: int = 0) -> int:
    """Extract a rounded integer from `raw`, falling back to `default`."""
    parsed = as_scalar_number(raw)
    return default if parsed is None else round(parsed)


def _require_metric_float(metrics: Mapping[str, object], key: str) -> float:
    """Extract a required numeric value from `metrics[key]`, failing closed."""
    raw_entry = metrics[key]
    if type(raw_entry) is not dict or "value" not in raw_entry:
        msg = f"Metric {key!r} is missing a numeric value entry"
        raise KeyError(msg)
    numeric = as_scalar_number(raw_entry["value"])
    if numeric is None:
        msg = f"Metric {key!r} has non-numeric value: {raw_entry['value']!r}"
        raise KeyError(msg)
    return numeric


def _code(value: object, *, raw: bool = False) -> str:
    """Wrap text in an inline `<code>` element, escaping unless `raw=True`."""
    inner = str(value) if raw else _escape_html(value)
    return f"<code>{inner}</code>"


def _badge(suffix: str, label_html: str, *, title: str = "") -> str:
    """Wrap escaped or structured text in a semantic `.badge` span."""
    title_attr = f' title="{_escape_html(title)}"' if title else ""
    return (
        f'<span class="badge badge-{_escape_html(suffix)}"{title_attr}>'
        f"{label_html}</span>"
    )


def _prov_tag(tag: object) -> str:
    """Render one `.prov-tag` provenance badge."""
    escaped_tag = _escape_html(tag)
    return f'<span class="prov-tag prov-tag-{escaped_tag}">{escaped_tag}</span>'


def _sub(body_html: str) -> str:
    """Wrap secondary caption HTML inside a `.kpi-sub` container."""
    return f'<div class="kpi-sub">{body_html}</div>'


def _render_flags_pills(flags: Sequence[str], recommended_scope: str) -> str:
    """Render pill badges for visible diagnostic warning flags."""
    visible = [
        f'<span class="flag-pill">{_escape_html(flag)}</span>'
        for flag in flags
        if flag not in HIDDEN_TABLE_FLAGS
        and not (flag in MEGA_ONLY_FLAGS and recommended_scope == "per_op")
    ]
    return " ".join(visible) if visible else '<span class="kpi-sub">none</span>'


def _html_table(headers: Sequence[str], rows_html: str, *, css_class: str = "") -> str:
    """Build an HTML table from explicit header labels (`#` marks tabular)."""
    cls_attr = f' class="{_escape_html(css_class)}"' if css_class else ""
    heads = "".join(
        f'<th class="tabular">{raw[:-1]}</th>'
        if raw.endswith("#")
        else f"<th>{raw}</th>"
        for raw in headers
    )
    return (
        f"<table{cls_attr}><thead><tr>{heads}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _kpi_card(label_html: str, value_html: str, sub_html: str) -> str:
    """Build one KPI summary card block."""
    return (
        f'<div class="card"><div class="kpi-label">{label_html}</div>'
        f'<div class="kpi-val tabular">{value_html}</div>{_sub(sub_html)}</div>'
    )


def _section_card(title_html: str, body_html: str, *, lead_html: str = "") -> str:
    """Wrap section content inside a standard report card."""
    return (
        f'<div class="card section-card"><h2 class="section-title">{title_html}'
        f"</h2>{lead_html}{body_html}</div>"
    )


def _report_header(heading: str, subtitle_html: str) -> str:
    """Render top title banner for single or sweep HTML reports."""
    return (
        f'<div class="report-header"><div><h1>{heading}</h1>'
        f'<div class="report-subtitle">{subtitle_html}</div></div></div>'
    )


class _SvgCanvas:
    """Lightweight builder for self-contained, HTML-escaped inline SVGs."""

    def __init__(
        self, width: float, height: float, *, with_border: bool = True
    ) -> None:
        """Initialize an SVG canvas with explicit `viewBox` dimensions."""
        svg_open = (
            f'<svg viewBox="0 0 {width:.0f} {height:.0f}" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg" role="img">'
        )
        self._parts = [svg_open]
        if with_border:
            self.rect(
                (0.0, 0.0, width, height),
                f'fill="{CHART_BG}" stroke="var(--border, #373B5E)" '
                'stroke-width="1" rx="6"',
            )

    def _elem(self, tag: str, attrs: str, body: str = "", title: str = "") -> None:
        """Append one XML element with optional inner text and tooltip."""
        title_xml = f"<title>{_escape_html(title)}</title>" if title else ""
        self._parts.append(f"<{tag} {attrs}>{title_xml}{body}</{tag}>")

    def rect(
        self, bounds: tuple[float, float, float, float], attrs: str, title: str = ""
    ) -> None:
        """Append a `<rect>` element at `(x, y, width, height)`."""
        x_pos, y_pos, width, height = bounds
        geom = (
            f'x="{x_pos:.1f}" y="{y_pos:.1f}" '
            f'width="{width:.1f}" height="{height:.1f}" {attrs}'
        )
        self._elem("rect", geom, title=title)

    def line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        attrs: str,
        title: str = "",
    ) -> None:
        """Append a `<line>` element connecting `start` and `end`."""
        geom = (
            f'x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}" {attrs}'
        )
        self._elem("line", geom, title=title)

    def whisker(
        self,
        span: tuple[float, float, float],
        color: str,
        floor_us: float,
        hi_us: float,
    ) -> None:
        """Append a horizontal uncertainty whisker with floor/upper attributes."""
        x_lo, x_hi, y_mid = span
        self._parts.append(
            f'<line class="uncertainty-whisker" data-floor="{floor_us:.2f}" '
            f'data-upper="{hi_us:.2f}" x1="{x_lo:.2f}" y1="{y_mid:.2f}" '
            f'x2="{x_hi:.2f}" y2="{y_mid:.2f}" stroke="{_escape_html(color)}" '
            'stroke-width="3"/>'
        )

    def circle(
        self, center: tuple[float, float], radius: float, attrs: str, title: str = ""
    ) -> None:
        """Append a `<circle>` element at `center` with `radius`."""
        geom = f'cx="{center[0]:.2f}" cy="{center[1]:.2f}" r="{radius:g}" {attrs}'
        self._elem("circle", geom, title=title)

    def text(
        self,
        pos: tuple[float, float],
        content: str,
        *,
        anchor: str = "start",
        font_size: int = SVG_MIN_FONT_PX,
        extra: str = "",
    ) -> None:
        """Append an escaped `<text>` element at `pos`."""
        fill = "" if "fill=" in extra else f' fill="{TEXT_COLOR}"'
        pad = f" {extra}" if extra else ""
        attrs = (
            f'x="{pos[0]:.1f}" y="{pos[1]:.1f}" '
            f'text-anchor="{anchor}" font-size="{font_size}"{fill}{pad}'
        )
        self._elem("text", attrs, body=_escape_html(content))

    def marker(
        self, shape: str, center: tuple[float, float], color: str, title: str
    ) -> None:
        """Append a circle, square, triangle, or diamond scope marker."""
        center_x, center_y = center
        radius = 7.0
        attrs = f'fill="{_escape_html(color)}" stroke="{CHART_BG}" stroke-width="2"'
        if shape == "circle":
            self.circle(center, radius, attrs, title)
            return
        if shape == "square":
            self.rect(
                (center_x - radius, center_y - radius, 2.0 * radius, 2.0 * radius),
                attrs,
                title,
            )
            return
        edge = radius + 1.0
        offsets = (
            ((0.0, -edge), (edge, radius), (-edge, radius))
            if shape == "triangle"
            else ((0.0, -edge), (edge, 0.0), (0.0, edge), (-edge, 0.0))
        )
        polygon_points = " ".join(
            f"{center_x + dx:.1f},{center_y + dy:.1f}" for dx, dy in offsets
        )
        self._elem("polygon", f'points="{polygon_points}" {attrs}', title=title)

    def render(self) -> str:
        """Finalize and return the complete inline `<svg>` document string."""
        return "\n".join([*self._parts, "</svg>"])


@dataclass(frozen=True, slots=True)
class _VmemTickBar:
    """Per-token-tick adaptive static VMEM and ring slot sizing for Chart 1."""

    token_count: int
    static_mib: float
    dense_slots: int
    expert_slots: int
    raw_max_vmem_tokens: int


def _vmem_tick_series(sweep: AdvisorSweepReport) -> list[_VmemTickBar]:
    """Evaluate adaptive ring slots and static VMEM for each Chart 1 token tick."""
    profile, chip_spec = sweep.model_profile, sweep.chip_spec
    if profile is None or chip_spec is None:
        expert = (
            sweep.dma_buffers if profile is not None and profile.num_experts > 0 else 0
        )
        return [
            _VmemTickBar(
                tok,
                sweep.vmem_static_mib,
                sweep.dma_buffers,
                expert,
                sweep.max_vmem_tokens,
            )
            for tok in VMEM_TOKEN_TICKS
        ]
    ref_workload = next(
        (e.workload for e in sweep.regimes if e.label.startswith("Decode Med")),
        next(
            (e.workload for e in sweep.regimes if e.phase == "decode"),
            sweep.regimes[0].workload if sweep.regimes else {},
        ),
    )
    recommended_layout = sweep.parallelism
    raw_budget = as_scalar_number(ref_workload.get("vmem_budget_mib"))
    usable_mib = chip_spec.vmem_usable_bytes / BYTES_PER_MIB
    vmem_budget = (
        raw_budget
        if raw_budget is not None
        else (
            sweep.vmem_budget_mib
            if abs(usable_mib - sweep.vmem_budget_mib) > HEADROOM_EPS_MIB
            or not sweep.regimes
            else None
        )
    )
    fit_headroom = (
        bool(ref_workload["vmem_fit_headroom"])
        if "vmem_fit_headroom" in ref_workload
        else abs(usable_mib - sweep.vmem_budget_mib) > HEADROOM_EPS_MIB
    )
    kv_split = _scalar_int(ref_workload.get("kv_split"), max(1, recommended_layout.dcp))
    moe_mode = MOE_MODE_MAP.get(ref_workload.get("moe_mode"), "masked_on_chip")
    bars: list[_VmemTickBar] = []
    for token_count in VMEM_TOKEN_TICKS:
        tick_workload = WorkloadSpec(
            phase="decode",
            batch=token_count * recommended_layout.dp,
            tp=recommended_layout.tp,
            ep=recommended_layout.ep,
            dp=recommended_layout.dp,
            pp=recommended_layout.pp,
            dcp=recommended_layout.dcp,
            pcp=recommended_layout.pcp,
            kv_split=kv_split,
            layer_groups=recommended_layout.layer_groups,
            moe_mode=moe_mode,
            vmem_alias=bool(ref_workload.get("vmem_alias", True)),
            vmem_fit_headroom=fit_headroom,
            vmem_budget_mib=vmem_budget,
            dma_buffers=sweep.dma_buffers,
            max_rows_per_call=0,
            batch_explicit=True,
        )
        estimates = evaluate_scopes(profile, tick_workload, chip_spec)
        ref = next(
            (s for s in estimates if s.scope == sweep.vmem_reference_scope),
            estimates[-1],
        )
        bars.append(
            _VmemTickBar(
                token_count,
                ref.vmem_static_mib,
                ref.selected_dense_slots,
                ref.selected_expert_slots,
                ref.max_vmem_tokens,
            )
        )
    return bars


def _vmem_cap_badge(
    sweep: AdvisorSweepReport, usable_mib: float, raw_max: int, *, row_capped: bool
) -> str:
    """Format the Chart 1 VMEM ceiling badge text."""
    cap_suffix = f", row cap < {raw_max} VMEM" if row_capped else ""
    fit_txt = f"{sweep.vmem_budget_mib:.1f} MiB"
    if abs(usable_mib - sweep.vmem_budget_mib) > HEADROOM_EPS_MIB:
        return (
            f"Usable VMEM Ceiling: {usable_mib:.1f} MiB | "
            f"Fit: {fit_txt} (T_max={sweep.max_vmem_tokens}{cap_suffix})"
        )
    return (
        f"Usable VMEM Ceiling: {fit_txt} "
        f"(T_max={sweep.max_vmem_tokens} tokens{cap_suffix})"
    )


def _vmem_bar_badge_and_color(
    tok: int,
    total_mib: float,
    sweep: AdvisorSweepReport,
    *,
    row_capped: bool,
) -> tuple[str, str]:
    """Compute Chart 1 bar badge text and accent color."""
    if sweep.max_vmem_tokens <= 0:
        return "VMEM OOM", "#EF4444"
    passes = max(1, math.ceil(tok / sweep.max_vmem_tokens))
    if passes == 1:
        return "1 pass", "#10B981"
    is_row_cap = row_capped and total_mib <= sweep.vmem_budget_mib + HEADROOM_EPS_MIB
    return (f"{passes}p cap" if is_row_cap else f"{passes} passes"), "#F59E0B"


def _render_vmem_svg(sweep: AdvisorSweepReport) -> str:
    """Render Chart 1 stacked VMEM token capacity vs usable VMEM budget."""
    chart_w, chart_h, pad_l, pad_t = 960.0, 310.0, 64.0, 42.0
    plot_w, plot_h = chart_w - pad_l - 32.0, chart_h - pad_t - 50.0
    slot_w = plot_w / len(VMEM_TOKEN_TICKS)
    bar_w = slot_w / 1.8
    res_bytes = sweep.vmem_breakdown_bytes["residual_buffer_bytes"]
    res_per_tok = res_bytes / BYTES_PER_MIB
    op_per_tok = (sweep.per_token_vmem_bytes - res_bytes) / BYTES_PER_MIB
    bars = _vmem_tick_series(sweep)
    raw_max = bars[0].raw_max_vmem_tokens
    row_capped = 0 < sweep.max_vmem_tokens < raw_max
    usable_mib = (
        sweep.chip_spec.vmem_usable_bytes / BYTES_PER_MIB
        if sweep.chip_spec is not None
        else sweep.vmem_budget_mib
    )
    max_total = max(
        bar.static_mib + (res_per_tok + op_per_tok) * bar.token_count for bar in bars
    )
    cap_mib = max(usable_mib * 1.15, sweep.vmem_budget_mib * 1.15, max_total * 1.08)

    canvas = _SvgCanvas(chart_w, chart_h, with_border=False)
    canvas.rect(
        (pad_l, pad_t, plot_w, plot_h),
        f'fill="{CHART_BG}" stroke="{CHART_GRID}" rx="6"',
    )
    x_right = pad_l + plot_w
    if usable_mib > sweep.vmem_budget_mib:
        y_use = pad_t + plot_h - min(1.0, usable_mib / cap_mib) * plot_h
        canvas.line(
            (pad_l, y_use),
            (x_right, y_use),
            'fill="none" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="4,4"',
        )
    y_fit = pad_t + plot_h - min(1.0, sweep.vmem_budget_mib / cap_mib) * plot_h
    canvas.line(
        (pad_l, y_fit),
        (x_right, y_fit),
        'fill="none" stroke="#EF4444" stroke-width="2.5" stroke-dasharray="8,5"',
    )
    cap_label = _vmem_cap_badge(sweep, usable_mib, raw_max, row_capped=row_capped)
    box_w = svg_text_width(cap_label, SVG_BADGE_FONT_PX) + 16.0
    box_x = max(pad_l, x_right - box_w - 4.0)
    canvas.rect(
        (box_x, pad_t - 26.0, box_w, 22.0),
        'fill="var(--card-inner, #1C1E32)" stroke="#EF4444" stroke-width="1.5" rx="4"',
    )
    canvas.text(
        (box_x + box_w / 2.0, pad_t - 11.0),
        cap_label,
        anchor="middle",
        font_size=SVG_BADGE_FONT_PX,
        extra='font-weight="700" fill="#FCA5A5"',
    )

    y_bottom = pad_t + plot_h
    for bar_index, bar in enumerate(bars):
        center_x = pad_l + (bar_index + 0.5) * slot_w
        tok = bar.token_count
        res_mib, op_mib = res_per_tok * tok, op_per_tok * tok
        total_mib = bar.static_mib + res_mib + op_mib
        static_height, residual_height, op_height = (
            (bar.static_mib / cap_mib) * plot_h,
            (res_mib / cap_mib) * plot_h,
            (op_mib / cap_mib) * plot_h,
        )
        static_y, residual_y, op_y = (
            y_bottom - static_height,
            y_bottom - static_height - residual_height,
            y_bottom - static_height - residual_height - op_height,
        )
        ring_txt = format_ring_slots(bar.dense_slots, bar.expert_slots)
        static_title = (
            f"T={tok}: Adaptive Weight Ring {ring_txt} & "
            f"Scratch = {bar.static_mib:.2f} MiB"
        )
        residual_title = f"T={tok}: On-Chip Residual Buffer [T, H] = {res_mib:.2f} MiB"
        op_title = (
            f"T={tok}: Shared Operator Scratchpad = "
            f"{op_mib:.2f} MiB (Total={total_mib:.1f} MiB)"
        )
        for seg_y, seg_h, color, seg_title in (
            (static_y, static_height, "#3b82f6", static_title),
            (residual_y, residual_height, "#10b981", residual_title),
            (op_y, op_height, "#f59e0b", op_title),
        ):
            canvas.rect(
                (center_x - bar_w / 2.0, seg_y, bar_w, seg_h),
                f'fill="{color}"',
                seg_title,
            )
        badge, fill_col = _vmem_bar_badge_and_color(
            tok, total_mib, sweep, row_capped=row_capped
        )
        bar_label = f"{total_mib:.1f}M ({badge})"
        label_y = max(pad_t + SVG_PAD_PX, op_y - 6.0)
        if (
            bar_index % 2 == 1
            and svg_text_width(bar_label, SVG_BADGE_FONT_PX) > slot_w - 8.0
        ):
            label_y = max(pad_t + SVG_PAD_PX + 14.0, label_y + 14.0)
        canvas.text(
            (center_x, label_y),
            bar_label,
            anchor="middle",
            font_size=SVG_BADGE_FONT_PX,
            extra=f'font-weight="600" fill="{fill_col}"',
        )
        canvas.text(
            (center_x, y_bottom + 20.0),
            f"T={tok}",
            anchor="middle",
            font_size=12,
            extra='font-weight="600"',
        )
    canvas.text(
        (pad_l + plot_w / 2.0, chart_h - 10.0),
        "Active Tokens per Step (T) | Blue: Adaptive Weight Ring & Scratch | "
        "Green: [T, H] Residual | Amber: Operator Scratchpad",
        anchor="middle",
        font_size=12,
        extra=f'fill="{MUTED}"',
    )
    return canvas.render()


def _row_limits(regime_entry: RegimeEntry) -> tuple[int, int, int]:
    """Return `(whole_step_max_tokens, single_layer_max_tokens, ridge_tokens)`."""
    ceils = {
        str(scope_dict["scope"]): _scalar_int(scope_dict.get("max_vmem_tokens"))
        for scope_dict in regime_entry.scopes
        if scope_dict.get("feasible")
    }
    return (
        ceils.get("whole_step", 0),
        ceils.get("single_layer", 0),
        round(_require_metric_float(regime_entry.metrics, "compute_bound_at")),
    )


def _scope_floor_us(regime_entry: RegimeEntry, scope_name: str) -> float:
    """Return the peak floor latency of `scope_name` when feasible, else 0.0."""
    for scope_dict in regime_entry.scopes:
        if scope_dict.get("scope") == scope_name and scope_dict.get("feasible"):
            return _scalar_float(scope_dict.get("step_bound_us"))
    return 0.0


def _tokens_to_x(
    token_count: float, x_start: float, plot_width_px: float, log_hi: float
) -> float:
    """Map tokens per core onto a horizontal log2 pixel coordinate."""
    return x_start + math.log2(max(1.0, float(token_count))) / log_hi * plot_width_px


def render_regime_dot_plot_svg(sweep: AdvisorSweepReport) -> str:
    """Render Chart 2: active tokens per core vs on-chip limits on a log2 axis."""
    regimes = sweep.regimes
    plot_w, row_h, top_base, mark_step = 520.0, 28.0, 34.0, 20.0
    row_limits = [_row_limits(entry) for entry in regimes]
    right_labels = [
        f"{e.recommended_scope}, {e.step_bound_us:,.0f} us" for e in regimes
    ]
    label_w = max((svg_text_width(e.label) for e in regimes), default=0.0)
    right_w = max((svg_text_width(text) for text in right_labels), default=0.0)
    x_start = SVG_PAD_PX + label_w + SVG_GAP_PX
    x_end = x_start + plot_w
    hi_tokens = max(
        [
            MIN_PLOT_TOKENS,
            *(e.active_tokens for e in regimes),
            *(tok for lims in row_limits for tok in lims if tok > 0),
        ]
    )
    ticks = tuple(tick for tick in SVG_TICKS if tick <= hi_tokens)
    log_hi = math.log2(max(hi_tokens, ticks[-1]))
    mark_labels = [f"{name} (this row's layout)" for name, _ in DOT_PLOT_MARKS]
    title = "Where each workload lands: tokens per core vs on-chip limits"
    width_px = max(
        x_end + SVG_GAP_PX + right_w + SVG_PAD_PX,
        2.0 * SVG_PAD_PX + svg_text_width(title, SVG_TITLE_FONT_PX),
        *(
            x_start + 18.0 + svg_text_width(mark_label) + SVG_PAD_PX
            for mark_label in mark_labels
        ),
        *(
            2.0 * SVG_PAD_PX + 24.0 + svg_text_width(f"{n}: {SCOPE_HINTS[n]}")
            for n in SCOPE_MARKS
        ),
    )
    top_y = top_base + mark_step * len(DOT_PLOT_MARKS) + 8.0
    bottom_y = top_y + row_h * len(regimes)
    legend_y = bottom_y + 60.0
    canvas = _SvgCanvas(width_px, legend_y + 22.0 * len(SCOPE_MARKS) + 12.0)
    canvas.text(
        (SVG_PAD_PX, 24.0),
        title,
        font_size=SVG_TITLE_FONT_PX,
        extra='font-weight="700"',
    )
    for idx, ((_, mark_color), mark_label) in enumerate(
        zip(DOT_PLOT_MARKS, mark_labels, strict=True)
    ):
        label_y = top_base + mark_step * (idx + 1)
        canvas.line(
            (x_start + 4.0, label_y - 11.0),
            (x_start + 4.0, label_y + 3.0),
            f'fill="none" stroke="{mark_color}" stroke-width="3"',
        )
        canvas.text((x_start + 14.0, label_y), mark_label, extra=f'fill="{mark_color}"')

    row_line_attr = (
        f'fill="none" stroke="{CHART_GRID}" stroke-width="1.5" stroke-dasharray="2,4"'
    )
    muted_attr = f'fill="{MUTED}"'
    for idx, (entry, limits) in enumerate(zip(regimes, row_limits, strict=True)):
        center_y = top_y + row_h * idx + row_h / 2.0
        for (mark_name, mark_color), limit_tokens in zip(
            DOT_PLOT_MARKS, limits, strict=True
        ):
            if limit_tokens > 0:
                mark_x = _tokens_to_x(limit_tokens, x_start, plot_w, log_hi)
                canvas.line(
                    (mark_x, center_y - 9.0),
                    (mark_x, center_y + 9.0),
                    f'fill="none" stroke="{mark_color}" stroke-width="3"',
                    title=f"{entry.label}: {mark_name} = {limit_tokens:,} tokens/core",
                )
        color, shape = SCOPE_MARKS.get(entry.recommended_scope, ("#cbd5e1", "circle"))
        canvas.line((x_start, center_y), (x_end, center_y), row_line_attr)
        canvas.text(
            (x_start - SVG_GAP_PX / 2.0, center_y + 4.5), entry.label, anchor="end"
        )
        core_note = f"core={entry.core_support}, core_ready={entry.core_ready_scope}"
        if entry.core_blockers:
            core_note = f"{core_note}, blocked by {', '.join(entry.core_blockers)}"
        tip = (
            f"{entry.label}: {entry.active_tokens} tokens/core, "
            f"AI {entry.step_arithmetic_intensity} FLOP/B, {entry.layout} -> "
            f"{entry.recommended_scope} ({entry.recommended_moe_mode}), "
            f"{entry.step_bound_us:.1f} us [{core_note}]"
        )
        x_tok = _tokens_to_x(entry.active_tokens, x_start, plot_w, log_hi)
        canvas.marker(shape, (x_tok, center_y), color, tip)
        canvas.text(
            (x_end + SVG_GAP_PX, center_y + 4.5), right_labels[idx], extra=muted_attr
        )

    axis_attr = f'fill="none" stroke="{MUTED}" stroke-width="1.5"'
    canvas.line((x_start, bottom_y), (x_end, bottom_y), axis_attr)
    for tick in ticks:
        tick_x = _tokens_to_x(tick, x_start, plot_w, log_hi)
        canvas.line((tick_x, bottom_y), (tick_x, bottom_y + 5.0), axis_attr)
        canvas.text(
            (tick_x, bottom_y + 22.0), f"{tick:,}", anchor="middle", extra=muted_attr
        )
    canvas.text(
        ((x_start + x_end) / 2.0, bottom_y + 44.0),
        "Active tokens per core per step (log scale)",
        anchor="middle",
    )
    for idx, (scope_name, (color, shape)) in enumerate(SCOPE_MARKS.items()):
        legend_y_pos = legend_y + 22.0 * idx
        canvas.marker(shape, (SVG_PAD_PX + 8.0, legend_y_pos - 4.5), color, scope_name)
        canvas.text(
            (SVG_PAD_PX + 24.0, legend_y_pos),
            f"{scope_name}: {SCOPE_HINTS[scope_name]}",
        )
    return canvas.render()


@dataclass(frozen=True, slots=True)
class _BandRow:
    """Normalized row payload for the shared latency-band whisker SVG renderer."""

    label: str
    floor_us: float
    upper_us: float
    color: str
    min_us_for_domain: float
    right_label: str
    per_op_us: float = 0.0
    achievable_us: float | None = None


def _latency_to_x(
    value_us: float, domain: tuple[float, float], x_start: float, plot_width_px: float
) -> float:
    """Map latency in microseconds onto a clamped horizontal log10 coordinate."""
    domain_min, domain_max = domain
    clamped = max(domain_min, min(domain_max, float(value_us)))
    log_min, log_max = math.log10(domain_min), math.log10(domain_max)
    frac = (math.log10(clamped) - log_min) / max(0.01, log_max - log_min)
    return x_start + max(0.0, min(1.0, frac)) * plot_width_px


def _decade_grid_ticks(domain_min: float, domain_max: float) -> list[float]:
    """Generate 1-2-5 decade grid ticks falling inside `[domain_min, domain_max]`."""
    ticks: list[float] = []
    decade = 10.0 ** math.floor(math.log10(max(1.0, domain_min)))
    while decade <= domain_max:
        ticks.extend(
            decade * mult
            for mult in DECADE_MULTS
            if domain_min <= decade * mult <= domain_max
        )
        decade *= 10.0
    if not ticks and domain_max >= domain_min:
        return [round((domain_min + domain_max) / 2.0, 1)]
    return ticks


def _render_latency_band_svg(
    rows: Sequence[_BandRow], title: str, legend: str, plot_width_px: float
) -> str:
    """Render horizontal log10 latency uncertainty whiskers with dynamic gridlines."""
    row_h, top_y, bottom_pad = 28.0, 62.0, 38.0
    label_w = max((svg_text_width(r.label) for r in rows), default=80.0)
    right_w = max((svg_text_width(r.right_label) for r in rows), default=0.0)
    x_start = SVG_PAD_PX + label_w + SVG_GAP_PX
    x_end = x_start + plot_width_px
    min_us = min(
        (r.min_us_for_domain for r in rows if r.min_us_for_domain > 0.0), default=50.0
    )
    max_us = max(
        [
            10.0,
            *(r.upper_us for r in rows),
            *(r.per_op_us for r in rows),
            *(r.achievable_us or 0.0 for r in rows),
        ]
    )
    domain_min = max(1.0, min_us * 0.75)
    domain = (domain_min, max(domain_min * 1.25, max_us * 1.15))
    width_px = max(
        x_end + SVG_GAP_PX + right_w + SVG_PAD_PX,
        2.0 * SVG_PAD_PX + svg_text_width(title, SVG_TITLE_FONT_PX),
        2.0 * SVG_PAD_PX + svg_text_width(legend),
    )
    bottom_y = top_y + row_h * len(rows)
    canvas = _SvgCanvas(width_px, bottom_y + bottom_pad)
    canvas.text(
        (SVG_PAD_PX, 24.0),
        title,
        font_size=SVG_TITLE_FONT_PX,
        extra='font-weight="700"',
    )
    muted_attr = f'fill="{MUTED}"'
    canvas.text((SVG_PAD_PX, 46.0), legend, extra=muted_attr)
    grid_attr = (
        f'fill="none" stroke="{CHART_GRID}" stroke-width="1.5" stroke-dasharray="3,3"'
    )
    for grid_us in _decade_grid_ticks(domain[0], domain[1]):
        grid_x = _latency_to_x(grid_us, domain, x_start, plot_width_px)
        canvas.line((grid_x, top_y), (grid_x, bottom_y), grid_attr)
        tick_label = (
            f"{grid_us / MS_PER_US:g}ms" if grid_us >= MS_PER_US else f"{grid_us:g}us"
        )
        canvas.text(
            (grid_x, bottom_y + 18.0), tick_label, anchor="middle", extra=muted_attr
        )

    row_attr = (
        f'fill="none" stroke="{CHART_GRID}" stroke-width="1.5" stroke-dasharray="2,4"'
    )
    for idx, row in enumerate(rows):
        center_y = top_y + row_h * idx + row_h / 2.0
        x_lo = _latency_to_x(row.floor_us, domain, x_start, plot_width_px)
        x_hi = max(x_lo, _latency_to_x(row.upper_us, domain, x_start, plot_width_px))
        canvas.line((x_start, center_y), (x_end, center_y), row_attr)
        canvas.text(
            (x_start - SVG_GAP_PX / 2.0, center_y + 4.5), row.label, anchor="end"
        )
        canvas.whisker((x_lo, x_hi, center_y), row.color, row.floor_us, row.upper_us)
        if row.per_op_us > 0.0:
            per_op_x = _latency_to_x(row.per_op_us, domain, x_start, plot_width_px)
            canvas.circle(
                (per_op_x, center_y),
                4.0,
                'fill="#EF4444" opacity="0.9"',
                title=(
                    f"per_op floor: {row.per_op_us:,.1f} us "
                    f"({row.per_op_us - row.floor_us:+,.1f} us)"
                ),
            )
        canvas.circle(
            (x_lo, center_y),
            5.0,
            f'fill="{_escape_html(row.color)}" stroke="{CHART_BG}" stroke-width="1.5"',
            title=f"{row.label} floor: {row.floor_us:,.1f} us",
        )
        if row.achievable_us is not None:
            achievable_x = _latency_to_x(
                row.achievable_us, domain, x_start, plot_width_px
            )
            canvas.marker(
                "diamond",
                (achievable_x, center_y),
                row.color,
                f"{row.label} achievable: {row.achievable_us:,.1f} us",
            )
        canvas.text(
            (x_end + SVG_GAP_PX, center_y + 4.5), row.right_label, extra=muted_attr
        )
    return canvas.render()


def render_step_band_svg(sweep: AdvisorSweepReport) -> str:
    """Render Chart 3 one-sided upward uncertainty whiskers across regimes."""
    rows = []
    for entry in sweep.regimes:
        hi_us = max(float(entry.step_bound_us), float(entry.step_upper_us))
        rows.append(
            _BandRow(
                label=entry.label,
                floor_us=float(entry.step_bound_us),
                upper_us=hi_us,
                color=SCOPE_MARKS.get(entry.recommended_scope, ("#38bdf8", "circle"))[
                    0
                ],
                min_us_for_domain=float(entry.step_bound_us),
                right_label=f"{entry.step_bound_us:,.1f}..{hi_us:,.1f} us",
                per_op_us=_scope_floor_us(entry, "per_op")
                if entry.recommended_scope != "per_op"
                else 0.0,
            )
        )
    return _render_latency_band_svg(
        rows,
        "One-sided upward step latency bands: Peak Floor to Peak Upper (us)",
        "dot: floor of the recommended scope | bar: up to the upper bound | "
        "red dot: per_op floor at the same inputs",
        520.0,
    )


def _render_single_scope_band_svg(advisor_report: AdvisorReport) -> str:
    """Render Chart 4 scope step latency comparison for a single `AdvisorReport`."""
    rows = []
    for scope_est in advisor_report.scopes:
        hi_us = max(scope_est.step_bound_us, scope_est.step_upper_us)
        min_dom = (
            scope_est.step_floor_strict_us
            if scope_est.step_floor_strict_us > 0.0
            else scope_est.step_bound_us
        )
        rows.append(
            _BandRow(
                label=scope_est.scope,
                floor_us=scope_est.step_bound_us,
                upper_us=hi_us,
                color=SCOPE_MARKS.get(scope_est.scope, ("#38bdf8", "circle"))[0],
                min_us_for_domain=min_dom,
                right_label=f"{scope_est.step_bound_us:,.1f}..{hi_us:,.1f} us",
                achievable_us=scope_est.step_achievable_us,
            )
        )
    return _render_latency_band_svg(
        rows,
        "Scope Step Latency Comparison: Peak Floor Band and Achievable Estimate (us)",
        "dot: peak floor | bar: peak upper | diamond: derated achievable",
        460.0,
    )


def _render_fact_table(rows: Sequence[tuple[str, str, str]]) -> str:
    """Render `(label, value_html, source_text)` rows into a `.facts-table`."""
    body = "".join(
        f"<tr><td>{_escape_html(label)}</td><td>{val_html}</td>"
        f'<td><span class="fact-src">{_escape_html(src)}</span></td></tr>'
        for label, val_html, src in rows
    )
    return f'<table class="facts-table">{body}</table>'


def _hardware_fact_rows(
    sweep: AdvisorSweepReport,
    chip_spec: chips.ChipSpec,
    chip_sources: Mapping[str, str],
) -> list[tuple[str, str, str]]:
    """Build `(label, value_html, source)` hardware rows for the top facts card."""
    vmem_mib = chip_spec.vmem_usable_bytes / BYTES_PER_MIB
    hbm_bw_gb_s = chip_spec.hbm_bw_bytes_per_s / 1e9
    fp8_tflops, bf16_tflops = (
        chip_spec.fp8_flops_per_s / 1e12,
        chip_spec.bf16_flops_per_s / 1e12,
    )
    ici_egress_gb_s = chip_spec.ici_egress_bw_bytes_per_s / 1e9
    ici_link_gb_s = chip_spec.ici_link_bw_bytes_per_s / 1e9
    spec_rows = [
        (
            "Usable VMEM / Core",
            f"<strong>{vmem_mib:.1f} MiB</strong>",
            "vmem_usable_bytes",
            "Hardware Spec",
        ),
        (
            "VMEM Fit Budget / Core",
            f"<strong>{sweep.vmem_budget_mib:.1f} MiB</strong>",
            "vmem_fit_headroom_bytes",
            "DERIVED: usable VMEM minus fit headroom",
        ),
        (
            "HBM Capacity / Core",
            f"<strong>{sweep.parallelism.hbm_capacity_gb:.1f} GB</strong>",
            "hbm_capacity_bytes",
            "Hardware Spec",
        ),
        (
            "HBM Bandwidth / Core",
            f"<strong>{hbm_bw_gb_s:,.0f} GB/s</strong>",
            "hbm_bw_bytes_per_s",
            "Hardware Spec",
        ),
        (
            "Peak Compute / Core",
            f"<strong>{fp8_tflops:,.0f} TFLOP/s (FP8) &middot; "
            f"{bf16_tflops:,.0f} TFLOP/s (BF16)</strong>",
            "fp8_flops_per_s",
            "Hardware Spec",
        ),
        (
            "ICI Interconnect Egress",
            f"<strong>{ici_egress_gb_s:,.0f} GB/s/core</strong>",
            "ici_egress_bw_bytes_per_s",
            "Hardware Spec",
        ),
        (
            "ICI Link (all-reduce transfer)",
            f"<strong>{ici_link_gb_s:,.0f} GB/s/link</strong>",
            "ici_link_bw_bytes_per_s",
            "Hardware Spec",
        ),
    ]
    rows = [
        (label, value_html, chip_sources.get(key, default_source))
        for label, value_html, key, default_source in spec_rows
    ]
    if load_cap_line := vector_register_load_cap_line(chip_spec):
        rows.insert(
            4,
            (
                "Vector Register Load Cap",
                f'<span class="fact-src">{_escape_html(load_cap_line)}</span>',
                "Vector register bandwidth check",
            ),
        )
    return rows


def _model_fact_rows(
    model_profile: ModelProfile, model_sources: Mapping[str, str]
) -> list[tuple[str, str, str]]:
    """Build `(label, value_html, source)` model geometry and dtype rows."""
    attention_desc = (
        f"attn_heads={model_profile.num_attention_heads}, "
        f"kv_heads={model_profile.num_kv_heads}, head_dim={model_profile.head_dim}"
    )
    if has_dsa_profile(model_profile):
        attention_desc += (
            f", top-k={model_profile.index_topk} "
            f"({model_profile.num_indexer_layers} indexer layers)"
        )
    has_moe = model_profile.num_experts > 0
    moe_desc = (
        f"experts={model_profile.num_experts} routed experts, "
        f"top_k={model_profile.top_k}, "
        f"intermediate={model_profile.moe_intermediate_size}"
        if has_moe
        else "Dense MLP"
    )
    dtype_warning = (
        f" {_badge('assumed', 'WARNING: default dtype (no quantization_config)')}"
        if model_profile.dtype_source.startswith("default")
        else ""
    )
    expert_part = (
        f"expert={_escape_html(model_profile.expert_dtype)}, " if has_moe else ""
    )
    dtype_html = (
        f"<strong>weight={_escape_html(model_profile.weight_dtype)}, {expert_part}"
        f"kv={_escape_html(model_profile.kv_dtype)}, "
        f"state={_escape_html(model_profile.state_dtype)}</strong>{dtype_warning}"
    )
    layers_html = (
        f"<strong>{model_profile.num_layers} layers &middot; "
        f"hidden={model_profile.hidden_size}</strong>"
    )
    spec_rows = (
        (
            "Source Configuration",
            f"<strong>{_escape_html(model_profile.model_type)}</strong>",
            "config",
            "HF config.json",
        ),
        ("Layers & Hidden Size", layers_html, "num_layers", "HF config.json"),
        (
            "Attention Geometry",
            f"<strong>{_escape_html(attention_desc)}</strong>",
            "attention_heads",
            "HF config.json",
        ),
        (
            "MoE / MLP Structure",
            f"<strong>{_escape_html(moe_desc)}</strong>",
            "moe",
            "Dense MLP",
        ),
        ("Weight & KV Dtypes", dtype_html, "dtypes", "HF config.json"),
    )
    return [
        (label, value_html, model_sources.get(key, default_source))
        for label, value_html, key, default_source in spec_rows
    ]


def _render_facts_ledger_card(sweep: AdvisorSweepReport) -> str:
    """Render the top Hardware & Model Specifications ledger HTML card."""
    if sweep.chip_spec is None or sweep.model_profile is None:
        missing = "chip_spec" if sweep.chip_spec is None else "model_profile"
        msg = f"sweep.{missing} is required to render the facts card"
        raise ValueError(msg)
    chip_sources = sweep.chip_sources or (sweep.chip_spec.sources or {})
    measured = [
        entry.metrics[key]
        for entry in sweep.regimes
        for key in CALIBRATED_PRIOR_KEYS
        if key in entry.metrics and entry.metrics[key]["tag"] == "MEASURED"
    ]
    dcn_provenance = (
        sweep.disagg_handoff.dcn_provenance
        if sweep.disagg_handoff is not None
        else "ASSUMED"
    )
    is_calibrated = bool(measured) or dcn_provenance == "MEASURED"
    if is_calibrated:
        source_label = str(measured[0]["source"]) if measured else "dcn_bw_gbps"
        cal_badge = (
            f"{_badge('measured', 'CALIBRATED TRACE')} "
            f'<span class="cal-source">Source: {_code(source_label)}</span>'
        )
    else:
        hint = (
            f"{CALIBRATE_CMD} --trace &lt;trace.xplane.pb&gt; "
            f"--model {_escape_html(sweep.model)} --out measurements.json"
        )
        cal_badge = (
            f"{_badge('assumed', 'UNCALIBRATED PRIORS')} "
            f'<span class="cal-hint">Calibrate via: {_code(hint, raw=True)}</span>'
        )
    measured_tail = (
        _prov_tag("MEASURED")
        if is_calibrated
        else f'<span class="kpi-sub">MEASURED: none yet (run {CALIBRATE_CMD})</span>'
    )
    legend = _prov_tag("DERIVED") + _prov_tag("ASSUMED") + measured_tail
    meshes = [
        f"{role}: {_code(sweep.submodule_labels[role])}"
        for role in ROLE_DISPLAY_ORDER
        if role in sweep.submodule_labels
    ]
    meshes_html = (
        _sub(f"Sublayer meshes: {' &middot; '.join(meshes)}") if meshes else ""
    )
    prior_items: list[tuple[str, str, object]] = []
    if sweep.regimes:
        first_metrics = sweep.regimes[0].metrics
        for required_prior_key in (
            "launch_gap_us",
            "collective_latency_us",
            "step_coverage",
        ):
            _require_metric_float(first_metrics, required_prior_key)
        launch_metric, collective_metric, coverage_metric = (
            first_metrics["launch_gap_us"],
            first_metrics["collective_latency_us"],
            first_metrics["step_coverage"],
        )
        prior_items.extend(
            [
                (
                    "Launch Gap:",
                    _escape_html(f"{launch_metric['value']} us"),
                    launch_metric["tag"],
                ),
                (
                    "Collective Latency:",
                    _escape_html(f"{collective_metric['value']} us/hop"),
                    collective_metric["tag"],
                ),
                (
                    "Step Coverage:",
                    _escape_html(f"{coverage_metric['value']}"),
                    coverage_metric["tag"],
                ),
            ]
        )
    if (handoff := sweep.disagg_handoff) is not None:
        handoff_low_ms, handoff_high_ms = handoff.dcn_xfer_bracket_ms
        handoff_desc = (
            f"{handoff.dcn_bw_gbps:.0f} Gbps/chip ({handoff.total_wire_mb:.1f} MB "
            f"&rarr; {handoff.dcn_xfer_ms:.2f} ms "
            f"[{handoff_low_ms:.2f}..{handoff_high_ms:.2f} ms])"
        )
        prior_items.append(
            (
                f"Host DCN Hand-Off ({handoff.isl} tok):",
                handoff_desc,
                handoff.dcn_provenance,
            )
        )
    empty_priors = (
        ""
        if sweep.regimes
        else '<div class="prior-item">'
        '<span class="prior-name">No regime rows evaluated.</span></div>'
    )
    priors_html = empty_priors + "".join(
        f'<div class="prior-item"><span class="prior-name">{label}</span>'
        f'<span class="prior-val">{_code(value_html, raw=True)}</span>'
        f"{_prov_tag(provenance_tag)}</div>"
        for label, value_html, provenance_tag in prior_items
    )
    hardware_table = _render_fact_table(
        _hardware_fact_rows(sweep, sweep.chip_spec, chip_sources)
    )
    model_table = _render_fact_table(
        _model_fact_rows(sweep.model_profile, sweep.model_sources)
    )
    ledger_sub = _sub(
        "Sourced specs below; values computed from them are tagged DERIVED "
        "in the KPI cards and in each row's number ledger"
    )
    return (
        '<div class="card facts-card"><div class="facts-header"><div>'
        '<h2 class="section-title">'
        "Hardware &amp; Model Architecture Specifications</h2>"
        f'{ledger_sub}</div><div class="prov-legend">{legend}</div></div>'
        f'{meshes_html}<div class="facts-grid">'
        f'<div class="facts-col"><h3>Hardware Specifications '
        f"({_code(sweep.chip)})</h3>{hardware_table}</div>"
        f'<div class="facts-col"><h3>Model Architecture Dimensions '
        f"({_code(sweep.model)})</h3>{model_table}</div></div>"
        '<div class="priors-bar"><div class="priors-header">'
        '<span class="priors-title">'
        "Runtime Priors &amp; Empirical Calibration Status</span>"
        f"<div>{cal_badge}</div></div>"
        f'<div class="priors-items">{priors_html}</div></div></div>'
    )


def _render_scope_verdict_tag(verdict_str: str, recommended_scope: str) -> str:
    """Render the primary verdict pill tag and optional tie/invalid subtitle."""
    if verdict_str.startswith("invalid_config("):
        reasons = verdict_str.removeprefix("invalid_config(").removesuffix(")").strip()
        return '<span class="verdict-tag verdict-invalid">INVALID CONFIG</span>' + _sub(
            _code(reasons)
        )
    if verdict_str.startswith("tie(measure:"):
        knob = verdict_str.removeprefix("tie(measure:").removesuffix(")").strip()
        return '<span class="verdict-tag verdict-tie">TIE</span>' + _sub(
            f"measure: {_code(knob)}"
        )
    scope_label = SCOPE_LABELS_HTML.get(recommended_scope, "SEPARATE OPS")
    tag_cls = (
        "verdict-fusion"
        if recommended_scope in MEGAKERNEL_LABELS
        else (
            "verdict-layer"
            if recommended_scope == "single_layer"
            else "verdict-separate"
        )
    )
    tag_html = f'<span class="verdict-tag {tag_cls}">{scope_label}</span>'
    if verdict_str.startswith("tie(objective_split:"):
        split = (
            verdict_str.removeprefix("tie(objective_split:").removesuffix(")").strip()
        )
        tag_html += _sub(f"layout split: {_code(split)}")
    return tag_html


def _render_core_readiness_sub(
    core_support: str,
    core_ready_scope: str,
    core_blockers: Sequence[str],
) -> str:
    """Render Core Readiness badge, ready scope, and blocker diagnostics."""
    support_esc = _escape_html(core_support)
    blockers_text = ", ".join(core_blockers)
    if core_support == "outside_core_limits" or blockers_text:
        shown_blockers = blockers_text or "none"
        tooltip = f"core_ready_scope={core_ready_scope}; blocked by: {shown_blockers}"
        badge_html = _badge(core_support, support_esc, title=tooltip)
        return _sub(
            f"Core: {badge_html} &middot; core_ready_scope={_code(core_ready_scope)}"
            f" &middot; blocked by: {_code(shown_blockers)}"
        )
    tooltip = f"core_ready_scope={core_ready_scope}"
    badge_html = _badge(core_support, support_esc, title=tooltip)
    return _sub(
        f"Core: {badge_html} &middot; core_ready_scope={_code(core_ready_scope)}"
    )


def _render_decision_verdict_cell(regime_entry: RegimeEntry) -> str:
    """Render Verdict & Action cell HTML for one Decision Table row."""
    cell = _render_scope_verdict_tag(
        regime_entry.verdict, regime_entry.recommended_scope
    )
    if not regime_entry.hbm_fits:
        cell += _sub('<span class="flag-pill">PHYSICAL OOM</span>')
    elif regime_entry.hbm_pool_overflow:
        pool_cap = _scalar_float(
            regime_entry.workload.get("hbm_utilization_cap"), HBM_UTILIZATION_CAP
        )
        cell += _sub(f"exceeds {round(pool_cap * 100.0)}% pool cap")
    if not regime_entry.verdict.startswith("invalid_config("):
        cell += _render_core_readiness_sub(
            regime_entry.core_support,
            regime_entry.core_ready_scope,
            regime_entry.core_blockers,
        )
    return cell


def _render_scopes_table(
    scope_rows: Sequence[Mapping[str, object]], *, single_mode: bool
) -> str:
    """Render evaluated scope rows for single-report or sweep-ledger tables."""
    if not scope_rows:
        return ""
    body_rows: list[str] = []
    for scope_dict in scope_rows:
        scope_name = _escape_html(scope_dict.get("scope"))
        feasible = "yes" if scope_dict.get("feasible") else "no"
        support = _escape_html(scope_dict.get("support", "baseline"))
        floor_us = _scalar_float(scope_dict.get("step_bound_us"))
        upper_us = max(
            floor_us, _scalar_float(scope_dict.get("step_upper_us"), floor_us)
        )
        strict_us = _scalar_float(scope_dict.get("step_floor_strict_us"), floor_us)
        achievable_us = _scalar_float(scope_dict.get("step_achievable_us"), floor_us)
        achievable_upper_us = _scalar_float(
            scope_dict.get("step_achievable_upper_us"), achievable_us
        )
        ring_str = _escape_html(
            format_ring_slots(
                _scalar_int(scope_dict.get("selected_dense_slots")),
                _scalar_int(scope_dict.get("selected_expert_slots")),
            )
        )
        launches, passes = (
            _scalar_int(scope_dict.get("ops_per_step")),
            _scalar_int(scope_dict.get("num_vmem_splits")),
        )
        lat_cells = "".join(
            f'<td class="tabular">{v:,.1f}</td>'
            for v in (floor_us, upper_us, strict_us, achievable_us)
        )
        if single_mode:
            max_tok = _scalar_int(scope_dict.get("max_vmem_tokens"))
            bind = _code(str(scope_dict.get("binding_unit") or ""))
            body_rows.append(
                f"<tr><td>{_badge(scope_name, scope_name)}</td><td>{feasible}</td>"
                f"<td>{_badge(support, support)}</td>{lat_cells}"
                f'<td class="tabular">{max_tok:,}</td>'
                f'<td class="tabular">{passes}</td>'
                f'<td class="tabular">{ring_str}</td>'
                f'<td class="tabular">{launches}</td><td>{bind}</td></tr>'
            )
        else:
            body_rows.append(
                f"<tr><td>{_code(scope_name, raw=True)}</td><td>{feasible}</td>"
                f'{lat_cells}<td class="tabular">{achievable_upper_us:,.1f}</td>'
                f"<td>{_code(support, raw=True)}</td>"
                f'<td class="tabular">{ring_str}</td>'
                f'<td class="tabular">{launches}</td>'
                f'<td class="tabular">{passes}</td></tr>'
            )
    header_spec: tuple[str, ...] = (
        (
            "Scope",
            "Feasible",
            "Support",
            "Peak Floor (us)#",
            "Peak Upper (us)#",
            "Strict Floor (us)#",
            "Derated Achievable (us)#",
            "Max VMEM Tokens#",
            "Passes#",
            "Ring",
            "Launches#",
            "Binding Unit",
        )
        if single_mode
        else (
            "Scope",
            "Feasible",
            "Peak Floor (us)#",
            "Peak Upper (us)#",
            "Strict floor (us)#",
            "Derated Achievable (us)#",
            "Derated Achievable upper (us)#",
            "Support",
            "Ring",
            "Launches#",
            "Passes#",
        )
    )
    return _html_table(
        header_spec, "".join(body_rows), css_class="" if single_mode else "ledger-table"
    )


def _render_metrics_table(metrics: Mapping[str, ProvenanceMetric]) -> tuple[str, str]:
    """Render provenance metric table HTML and tag-count summary."""
    counts = dict.fromkeys(PROVENANCE_TAGS, 0)
    rows: list[str] = []
    for name, metric in metrics.items():
        tag = str(metric["tag"])
        counts[tag] = counts.get(tag, 0) + 1
        band = metric["band"]
        band_str = (
            f"[{band[0]}, {band[1]}]" if len(band) >= MIN_PROVENANCE_BAND_LEN else "[]"
        )
        rows.append(
            f"<tr><td>{_code(name)}</td><td>{_prov_tag(tag)}</td>"
            f'<td class="tabular">{_escape_html(metric["value"])}</td>'
            f"<td>{_escape_html(metric['unit'])}</td>"
            f'<td class="tabular">{_escape_html(band_str)}</td>'
            f"<td>{_escape_html(metric['source'])}</td></tr>"
        )
    summary = ", ".join(f"{counts[t]} {t.lower()}" for t in PROVENANCE_TAGS)
    return _html_table(
        ("Metric", "Tag", "Value#", "Unit", "Band#", "Source"),
        "".join(rows),
        css_class="ledger-table",
    ), summary


def _render_decision_row(regime_entry: RegimeEntry) -> str:
    """Render one Decision Table row plus its collapsed provenance ledger row."""
    scope_esc = _escape_html(regime_entry.recommended_scope)
    hi_us = max(regime_entry.step_bound_us, regime_entry.step_upper_us)
    metrics = regime_entry.metrics
    (
        throughput,
        gain_val,
        physics_gain_us,
        dispatch_gain_us,
        physics_gain_pct,
        break_even_us,
    ) = (
        _require_metric_float(metrics, key)
        for key in (
            "throughput_tok_s_per_chip",
            "fusion_gain_max",
            "compiler_proof_gain_us",
            "dispatch_bubble_gain_us",
            "compiler_proof_gain_pct",
            "break_even_host_head_delta_us",
        )
    )
    rec_scope_dict = recommended_scope_dict(regime_entry)
    achievable_us = _scalar_float(
        rec_scope_dict.get("step_achievable_us"),
        regime_entry.step_achievable_us,
    )
    achievable_upper_us = _scalar_float(
        rec_scope_dict.get("step_achievable_upper_us"),
        regime_entry.step_achievable_upper_us,
    )
    ring = format_ring_slots(
        _scalar_int(rec_scope_dict.get("selected_dense_slots")),
        _scalar_int(rec_scope_dict.get("selected_expert_slots")),
    )
    ring_sub = _sub(f"ring={_code(ring)}") if ring != "-" else ""
    gain_info = (
        f"physics: {physics_gain_us:+.1f} us ({physics_gain_pct:+.1f}%) | "
        f"dispatch: {dispatch_gain_us:+.1f} us"
    )
    gain_str = f"+{gain_val * 100.0:.1f}%" if gain_val > 0.0 else "+0.0%"
    gain_cls = "text-gain" if gain_val > 0.0 else "text-muted"
    gain_sub = gain_info
    break_even_html = (
        f"<br/>break-even host-head: +{break_even_us:,.1f} us"
        if regime_entry.recommended_scope == "per_op" and break_even_us > 0.0
        else ""
    )
    gain_basis_text = _escape_html(gain_basis_label(regime_entry.gain_basis))
    flags_html = _render_flags_pills(regime_entry.flags, regime_entry.recommended_scope)
    scope_cell = (
        f"{_badge(scope_esc, scope_esc)}"
        f"{_sub(_code(regime_entry.recommended_moe_mode))}{ring_sub}"
    )
    main_row = (
        f"<tr><td><strong>{_escape_html(regime_entry.label)}</strong></td>"
        f"<td>{_code(regime_entry.layout)}</td>"
        f'<td class="tabular">{regime_entry.active_tokens:,}</td>'
        f'<td class="tabular">{regime_entry.step_arithmetic_intensity:.2f}</td>'
        f'<td class="tabular">{regime_entry.num_vmem_splits}</td>'
        f"<td>{scope_cell}</td>"
        f'<td class="tabular"><strong>{regime_entry.step_bound_us:,.1f} .. '
        f'{hi_us:,.1f}</strong><div class="kpi-sub tabular" '
        f'style="text-align: right;">derated achievable: '
        f"{achievable_us:,.1f}..{achievable_upper_us:,.1f} us</div></td>"
        f'<td class="tabular">{throughput:,.1f}</td>'
        f'<td class="tabular {gain_cls}">{gain_str}<div class="kpi-sub tabular" '
        f'style="text-align: right;">{gain_sub}<br/>basis: {gain_basis_text}'
        f"{break_even_html}</div></td>"
        f"<td>{_render_decision_verdict_cell(regime_entry)}</td>"
        f"<td>{flags_html}</td><td>{_escape_html(regime_entry.rationale)}</td></tr>"
    )
    scope_tbl = _render_scopes_table(regime_entry.scopes, single_mode=False)
    metric_tbl, summary = _render_metrics_table(metrics)
    ledger_row = (
        f'<tr class="ledger-row"><td colspan="{DECISION_TABLE_COLS}">'
        '<details class="prov-details">'
        f"<summary>Numbers behind this row: {_escape_html(summary)}</summary>"
        f"{scope_tbl}{metric_tbl}</details></td></tr>"
    )
    return main_row + ledger_row


def _render_layer_group_search_card(search: LayerGroupSearchReport | None) -> str:
    """Render an HTML card for `LayerGroupSearchReport` when present."""
    if search is None or not search.candidates:
        return ""
    rows: list[str] = []
    for candidate in search.candidates:
        best_mark = (
            " <strong>[best]</strong>"
            if search.best is not None
            and candidate.layer_groups == search.best.layer_groups
            else ""
        )
        rows.append(
            f"<tr><td>{_escape_html(candidate.label)}{best_mark}</td>"
            f'<td class="tabular">{candidate.num_groups}</td>'
            f"<td>{'yes' if candidate.feasible else 'no'}</td>"
            f'<td class="tabular">{candidate.step_bound_us:.1f}</td>'
            f'<td class="tabular">{candidate.step_upper_us:.1f}</td>'
            f'<td class="tabular">{candidate.step_achievable_us:.1f}</td></tr>'
        )
    best_line = ""
    if (best := search.best) is not None:
        listed = (
            f"; gap vs listed grouping: {search.gap_vs_listed_us:+.1f} us "
            f"({search.gap_vs_listed_pct:+.2f}%)"
            if search.gap_vs_listed_us is not None
            and search.gap_vs_listed_pct is not None
            else ""
        )
        best_line = (
            '<p class="kpi-sub">Lowest-bound feasible partition: '
            f"<strong>{_escape_html(best.label)}</strong> "
            f"({best.step_bound_us:.1f} us vs <code>per_op</code> "
            f"{search.per_op_step_bound_us:.1f} us, "
            f"{search.gap_vs_per_op_us:+.1f} us / "
            f"{search.gap_vs_per_op_pct:+.2f}%{listed}).</p>"
        )
    table_html = _html_table(
        (
            "Partition",
            "Calls#",
            "Feasible",
            "Peak Floor (us)#",
            "Peak Upper (us)#",
            "Derated Achievable (us)#",
        ),
        "".join(rows),
    )
    title = (
        f"7. Layer-Group Partition Search ({_escape_html(search.phase)} on "
        f"{_escape_html(search.layout)}, {search.active_tokens} active tokens)"
    )
    return _section_card(
        title,
        table_html,
        lead_html=f'<p class="kpi-sub">{_escape_html(search.note)}</p>{best_line}',
    )


def _sweep_executive_kpi_card(sweep: AdvisorSweepReport) -> tuple[str, str, str]:
    """Build the Executive Scope & Core Readiness KPI card for a regime sweep."""
    if not sweep.regimes:
        return (
            "Executive Scope &amp; Core Readiness",
            "No regimes",
            "No regime rows evaluated.",
        )
    ref_entry = next(
        (e for e in sweep.regimes if e.label.startswith("Decode Small")),
        next(
            (e for e in sweep.regimes if e.label.startswith("Decode Med")),
            next(
                (e for e in sweep.regimes if e.phase == "decode"),
                sweep.regimes[0],
            ),
        ),
    )
    if ref_entry.verdict.startswith("invalid_config("):
        return (
            "Executive Scope &amp; Core Readiness",
            '<span class="verdict-tag verdict-invalid">INVALID CONFIG</span>',
            _code(ref_entry.verdict),
        )
    valid_regimes = [
        e for e in sweep.regimes if not e.verdict.startswith("invalid_config(")
    ]
    fused_count = sum(1 for e in valid_regimes if e.recommended_scope != "per_op")
    baseline_count = len(valid_regimes) - fused_count
    ready_count = sum(
        1 for e in valid_regimes if e.core_support == "within_core_limits"
    )
    scope_badge = _badge(
        ref_entry.recommended_scope,
        _escape_html(ref_entry.recommended_scope.upper()),
    )
    blockers_text = ", ".join(ref_entry.core_blockers)
    core_tooltip = (
        f"core_ready_scope={ref_entry.core_ready_scope}; blocked by: {blockers_text}"
        if ref_entry.core_support == "outside_core_limits" or blockers_text
        else f"core_ready_scope={ref_entry.core_ready_scope}"
    )
    core_badge = _badge(
        ref_entry.core_support,
        _escape_html(ref_entry.core_support),
        title=core_tooltip,
    )
    blocker_clause = (
        f" &middot; blocked by: {_code(blockers_text or 'none')}"
        if ref_entry.core_support == "outside_core_limits" or blockers_text
        else ""
    )
    sub_html = (
        f"{_escape_html(ref_entry.label)}: "
        f"core_ready_scope={_code(ref_entry.core_ready_scope)}"
        f"{blocker_clause}<br/>"
        f"Sweep: {ready_count}/{fused_count} fused core-ready &middot; "
        f"{baseline_count} baseline per_op"
    )
    return (
        "Executive Scope &amp; Core Readiness",
        f"{scope_badge} {core_badge}",
        sub_html,
    )


def _render_sweep_kpi_grid(sweep: AdvisorSweepReport) -> str:
    """Render the top KPI grid for a multi-regime sweep HTML report."""
    recommended_layout = sweep.parallelism
    rec_lbl = _escape_html(recommended_layout.label())
    mode_src = _escape_html(
        recommended_layout.source
        or ("auto-derived" if recommended_layout.auto_selected else "manual")
    )
    extra = (
        _sub(
            f"Prefill: {_escape_html(prefill_layout.label())}"
            + (
                f" ({_escape_html(prefill_layout.source)})"
                if prefill_layout.source
                else ""
            )
        )
        if (prefill_layout := sweep.prefill_parallelism) is not None
        else ""
    )
    if (saturation_layout := sweep.saturation_parallelism) is not None:
        extra += _sub(f"Saturated Decode: {_escape_html(saturation_layout.label())}")
    res_kib = sweep.vmem_breakdown_bytes["residual_buffer_bytes"] // 1024
    static = (
        f"{sweep.vmem_static_mib:.1f} / {sweep.vmem_budget_mib:.1f} MiB "
        f"({sweep.dma_buffers}-slot DMA)"
    )
    derived_tag = _prov_tag("DERIVED")
    card_specs = (
        _sweep_executive_kpi_card(sweep),
        (
            "Target Chip &amp; Mesh",
            f"{_escape_html(sweep.chip)} &middot; {rec_lbl}",
            (
                f"Decode slice: {_escape_html(recommended_layout.topology)} "
                f"({mode_src}){extra}"
            ),
        ),
        (
            f"Ideal-Kernel VMEM Ceiling {derived_tag}",
            f"{sweep.max_vmem_tokens:,} tokens/core",
            "Upper bound on rows per call, capped by max_rows_per_call; "
            f"not a served limit on its own. Static: {static}; decode layout {rec_lbl}",
        ),
        (
            f"Per-Token VMEM Slope {derived_tag}",
            f"{sweep.per_token_vmem_bytes / 1024.0:.1f} KiB/tok",
            f"[T, H] Residual {res_kib}K + Operator Scratchpad",
        ),
        (
            f"Compute-Bound Point {derived_tag}",
            f"{sweep.ridge_dense_flop_per_byte:.1f} FLOP/B",
            f"&asymp; {sweep.ridge_tokens} tokens/core/step on the decode layout; "
            "each row's own point is in Chart 2 and its ledger",
        ),
    )
    cards = "".join(_kpi_card(*spec) for spec in card_specs)
    return f'<div class="grid-kpi">{cards}</div>'


def _hint_list(hints: Mapping[str, str]) -> str:
    """Render an HTML unordered list of bold code keys and short descriptions."""
    items = "".join(
        f"<li><strong>{_code(k)}</strong>: {_escape_html(v)}.</li>"
        for k, v in hints.items()
    )
    return f"<ul>{items}</ul>"


def _render_sweep_html(sweep: AdvisorSweepReport) -> str:
    """Render a self-contained static HTML/SVG report for `AdvisorSweepReport`."""
    recommended_layout = sweep.parallelism
    mode_src = recommended_layout.source or (
        "auto-derived" if recommended_layout.auto_selected else "manual"
    )
    sub = sweep.submodule_labels.get("decode", "")
    sub_inline = f" &middot; {_code(sub)}" if sub else ""
    decision_hdr: tuple[str, ...] = (
        "Request Regime",
        "Layout",
        "Tokens (T)#",
        "Arithmetic Intensity (FLOP/B)#",
        "Passes#",
        "Scope &amp; MoE",
        "Peak Floor Band (eta=1.0, us)#",
        "Throughput (tok/s/chip)#",
        "Gain vs per_op#",
        "Verdict &amp; Action",
        "Flags",
        "Mechanical Rationale",
    )
    decision_tbl = _html_table(
        decision_hdr, "".join(_render_decision_row(e) for e in sweep.regimes)
    )
    rules_html = "".join(f"<li>{_escape_html(r)}</li>" for r in sweep.tailoring_guide)
    glossary = (
        "<p><strong>Topology terms:</strong> <code>core</code> = 1 logical TPU core; "
        "<code>chip</code> = 1 physical TPU package (2 cores on tpu7x, 1 core on v6e); "
        "<code>host</code> = 1 CPU VM node attached to 4 chips (8 cores on tpu7x; "
        "single-host is <code>2x2x1</code>); <code>slice</code> = 1 ICI-connected "
        "multi-chip torus; <code>instance</code> = 1 model replica (<code>1x</code> "
        "counts instances).</p><p>Scope: where the kernel boundary falls.</p>"
        f"{_hint_list(SCOPE_HINTS)}<p>MoE mode: where tokens are grouped by expert and "
        f"where the data lives.</p>{_hint_list(MOE_MODE_SHORT_HINTS)}"
    )
    step_lead = (
        '<p class="kpi-sub">Step bounds are hardware floors, not predictions: across '
        "analytical envelope fixtures they sit below derated achievable step "
        "estimates. eta is the efficiency factor applied to peak hardware rates; "
        "eta=1.0 means peak rates. A small margin to the runner-up is a close call; "
        "A/B the two scopes before building.</p>"
    )
    rec_lbl = _escape_html(recommended_layout.label())
    sections = "".join(
        (
            _section_card(
                f"1. VMEM Capacity &amp; Single-Pass Token Ceiling "
                f"(decode layout {rec_lbl})",
                _render_vmem_svg(sweep),
            ),
            _section_card(
                "2. Where Each Workload Lands (Tokens per Core vs On-Chip Limits)",
                render_regime_dot_plot_svg(sweep),
            ),
            _section_card(
                "3. Step Latency Floor vs Upper Uncertainty Band",
                render_step_band_svg(sweep),
            ),
            _section_card("4. How to Read the Scope and MoE Mode", glossary),
            _section_card(
                "5. Request Regime Recommendations", decision_tbl, lead_html=step_lead
            ),
            _section_card(
                f"6. How to Tailor to Your Request Mix "
                f"(thresholds from the decode layout {rec_lbl})",
                f"<ul>{rules_html}</ul>",
            ),
            _render_layer_group_search_card(sweep.layer_group_search),
        )
    )
    model_esc, chip_esc = _escape_html(sweep.model), _escape_html(sweep.chip)
    subtitle = (
        f"Model: <strong>{model_esc}</strong> &middot; "
        f"Chip: <strong>{chip_esc}</strong> &middot; "
        f"Slice: <strong>{_escape_html(recommended_layout.topology)}</strong> &middot; "
        f"Parallelism: <strong>{rec_lbl}</strong>{sub_inline} "
        f"({_escape_html(mode_src)})"
    )
    header_spec = _report_header(
        "Megakernel Architecture &amp; Regime Advisor", subtitle
    )
    ledger = _render_facts_ledger_card(sweep)
    grid = _render_sweep_kpi_grid(sweep)
    body = f'<div class="container">{header_spec}{ledger}{grid}{sections}</div>'
    return _wrap_html_document(
        f"Megakernel Advisor Report - {model_esc} ({chip_esc})", body
    )


def _render_single_report_html(advisor_report: AdvisorReport) -> str:
    """Render a self-contained static HTML/SVG audit report for `AdvisorReport`."""
    pool_pct = round(advisor_report.hbm_utilization_cap * 100.0)
    hbm_note = (
        "physical OOM (exceeds 1.00x HBM capacity)"
        if not advisor_report.hbm_fits
        else (
            f"hbm_pool_overflow (exceeds {pool_pct}% pool cap)"
            if advisor_report.hbm_pool_overflow
            else "fits physical HBM and pool cap"
        )
    )
    verdict_html = _render_scope_verdict_tag(
        advisor_report.verdict, advisor_report.recommended_scope
    )
    oom_pill = (
        ' <span class="flag-pill">PHYSICAL OOM</span>'
        if not advisor_report.hbm_fits
        else ""
    )
    flags_html = _render_flags_pills(
        advisor_report.flags, advisor_report.recommended_scope
    )
    metric_tbl, summary = _render_metrics_table(advisor_report.metrics)
    scope_dicts = [asdict(s) for s in advisor_report.scopes]
    assumptions_card = ""
    if advisor_report.unresolved_assumptions:
        assumption_rows = "".join(
            f"<tr><td>{_code(x.get('name'))}</td>"
            f'<td class="tabular">{_escape_html(x.get("value", "-"))}</td>'
            f'<td class="tabular">{_escape_html(x.get("band", []))}</td>'
            f"<td>{_escape_html(x.get('unit'))}</td>"
            f"<td>{_escape_html(x.get('affects', 'gain'))}</td>"
            f"<td>{_escape_html(x.get('how_to_settle'))}</td></tr>"
            for x in advisor_report.unresolved_assumptions
        )
        assumptions_card = _section_card(
            "Unresolved Assumptions",
            _html_table(
                ("Input", "Assumed#", "Band#", "Unit", "Affects", "How to Settle"),
                assumption_rows,
                css_class="ledger-table",
            ),
        )
    hbm_val = (
        f"{advisor_report.hbm_used_gb:.2f} / "
        f"{advisor_report.hbm_capacity_gb:.1f} GB/core"
    )
    is_invalid = advisor_report.verdict.startswith("invalid_config(")
    scope_kpi_val = (
        '<span class="verdict-tag verdict-invalid">INVALID CONFIG</span>'
        if is_invalid
        else _badge(
            advisor_report.recommended_scope,
            _escape_html(advisor_report.recommended_scope.upper()),
        )
    )
    bind_label = advisor_report.binding_unit
    blockers_text = ", ".join(advisor_report.core_blockers)
    core_tooltip = (
        f"core_ready_scope={advisor_report.core_ready_scope}; "
        f"blocked by: {blockers_text or 'none'}"
        if advisor_report.core_support == "outside_core_limits" or blockers_text
        else f"core_ready_scope={advisor_report.core_ready_scope}"
    )
    core_kpi_val = _badge(
        advisor_report.core_support,
        _escape_html(advisor_report.core_support),
        title=core_tooltip,
    )
    core_blockers_sub = (
        f" &middot; blocked by: {_code(blockers_text or 'none')}"
        if advisor_report.core_support == "outside_core_limits" or blockers_text
        else ""
    )
    kpi_specs = (
        (
            "Recommended Scope",
            scope_kpi_val,
            f"MoE: {_code(advisor_report.recommended_moe_mode)} &middot; "
            f"build={advisor_report.build_megakernel} &middot; "
            f"rank={_escape_html(advisor_report.ranking_metric)} "
            f"(bind={_escape_html(bind_label)})",
        ),
        (
            "Core Readiness",
            core_kpi_val,
            f"core_ready={advisor_report.core_ready} &middot; "
            f"core_ready_scope={_code(advisor_report.core_ready_scope)}"
            f"{core_blockers_sub}",
        ),
        ("HBM Footprint", hbm_val, _escape_html(hbm_note)),
        (
            "VMEM Budget &amp; Slope",
            f"{advisor_report.vmem_budget_mib:.1f} MiB",
            f"{advisor_report.dma_buffers}-slot DMA &middot; "
            f"{advisor_report.per_token_vmem_bytes / 1024.0:.1f} KiB/tok",
        ),
        (
            "Roofline Ridge &amp; Intensity",
            f"{advisor_report.ridge_dense_flop_per_byte:.1f} FLOP/B",
            f"T_ridge={advisor_report.ridge_tokens} &middot; "
            "Arithmetic Intensity="
            f"{advisor_report.step_arithmetic_intensity:.2f} FLOP/B",
        ),
    )
    kpis = "".join(_kpi_card(*spec) for spec in kpi_specs)
    model_esc, chip_esc = (
        _escape_html(advisor_report.model),
        _escape_html(advisor_report.chip),
    )
    subtitle = (
        f"Model: <strong>{model_esc}</strong> &middot; "
        f"Chip: <strong>{chip_esc}</strong> &middot; "
        f"Phase: <strong>{_escape_html(advisor_report.phase)}</strong> "
        f"(Tokens={advisor_report.active_tokens}, "
        f"Context={advisor_report.effective_context}) "
        f"&middot; Evidence: {_code(advisor_report.evidence)} "
        f"({_escape_html(advisor_report.confidence)})"
    )
    basis_sub = _sub(
        f"Coverage: {_escape_html(advisor_report.coverage)} &middot; "
        f"Basis: {_escape_html(gain_basis_label(advisor_report.gain_basis))}"
    )
    core_sub = (
        ""
        if is_invalid
        else _render_core_readiness_sub(
            advisor_report.core_support,
            advisor_report.core_ready_scope,
            advisor_report.core_blockers,
        )
    )
    verdict_body = (
        f"<div>{verdict_html}{oom_pill}</div>{core_sub}"
        f"<p>{_escape_html(advisor_report.rationale)}</p>{basis_sub}"
        f'<div style="margin-top: 8px;">Flags: {flags_html}</div>'
    )
    sections = "".join(
        (
            _section_card("Verdict &amp; Rationale", verdict_body),
            _section_card(
                "1. Scope Step Latency Comparison",
                _render_single_scope_band_svg(advisor_report),
            ),
            _section_card(
                "2. Evaluated Scope Breakdown",
                _render_scopes_table(scope_dicts, single_mode=True),
            ),
            _section_card(
                f"3. Provenance Metrics ({_escape_html(summary)})", metric_tbl
            ),
            assumptions_card,
        )
    )
    header_spec = _report_header("Megakernel Architecture &amp; Scope Audit", subtitle)
    body = (
        f'<div class="container">{header_spec}'
        f'<div class="grid-kpi">{kpis}</div>{sections}</div>'
    )
    return _wrap_html_document(
        f"Megakernel Scope Audit - {model_esc} ({chip_esc})", body
    )


def _wrap_html_document(escaped_title: str, body_html: str) -> str:
    """Wrap body HTML in a standalone zero-script HTML5 document with CSS."""
    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8"/>\n'
        f"<title>{escaped_title}</title>\n<style>\n{load_report_css()}\n</style>\n"
        f"</head>\n<body>\n{body_html}\n</body>\n</html>\n"
    )


def render_advisor_html(report_data: AdvisorReport | AdvisorSweepReport) -> str:
    """Render a self-contained, zero-script HTML/SVG audit report."""
    match report_data:
        case AdvisorSweepReport() as sweep:
            return _render_sweep_html(sweep)
        case AdvisorReport() as advisor_report:
            return _render_single_report_html(advisor_report)
        case _:
            msg = (
                "Expected AdvisorReport or AdvisorSweepReport, "
                f"got {type(report_data).__name__}"
            )
            raise TypeError(msg)


def write_advisor_html(
    report_data: AdvisorReport | AdvisorSweepReport, path: str | Path
) -> Path:
    """Write a self-contained static HTML/SVG audit report to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_advisor_html(report_data), encoding="utf-8")
    return output_path


def _bucket_sort_key(regime_entry: RegimeEntry) -> tuple[int, str, int, int]:
    """Return deterministic sort key ordering regimes for bucket coalescing."""
    return (
        PHASE_SORT_ORDER.get(regime_entry.phase, UNKNOWN_PHASE_SORT_ORDER),
        regime_entry.layout,
        regime_entry.active_tokens,
        regime_entry.effective_context,
    )


def build_bucket_schedule(sweep: AdvisorSweepReport) -> BucketScheduleReport:
    """Coalesce valid, HBM-fitting regimes into an offline bucket-schedule report.

    Rows sharing `(phase, layout, active_tokens)` with distinct scopes,
    verdicts, or core-readiness status across `effective_context` remain
    separate intervals distinguished by their `effective_context`.
    """
    valid_regimes = [
        entry
        for entry in sweep.regimes
        if entry.hbm_fits and not entry.verdict.startswith("invalid_config(")
    ]
    entries: list[BucketScheduleEntry] = []
    for entry in sorted(valid_regimes, key=_bucket_sort_key):
        if (
            entries
            and entries[-1].phase == entry.phase
            and entries[-1].layout == entry.layout
            and entries[-1].recommended_scope == entry.recommended_scope
            and entries[-1].recommended_moe_mode == entry.recommended_moe_mode
            and entries[-1].verdict == entry.verdict
            and entries[-1].core_support == entry.core_support
            and entries[-1].core_ready_scope == entry.core_ready_scope
            and entries[-1].core_blockers == entry.core_blockers
        ):
            prev = entries[-1]
            entries[-1] = replace(
                prev,
                max_active_tokens=max(prev.max_active_tokens, entry.active_tokens),
                effective_context=max(prev.effective_context, entry.effective_context),
                max_vmem_tokens=min(prev.max_vmem_tokens, entry.max_vmem_tokens),
                num_vmem_splits=max(prev.num_vmem_splits, entry.num_vmem_splits),
                step_bound_us=entry.step_bound_us,
            )
        else:
            entries.append(
                BucketScheduleEntry(
                    phase=entry.phase,
                    layout=entry.layout,
                    min_active_tokens=entry.active_tokens,
                    max_active_tokens=entry.active_tokens,
                    effective_context=entry.effective_context,
                    recommended_scope=entry.recommended_scope,
                    recommended_moe_mode=entry.recommended_moe_mode,
                    max_vmem_tokens=entry.max_vmem_tokens,
                    num_vmem_splits=entry.num_vmem_splits,
                    step_bound_us=entry.step_bound_us,
                    verdict=entry.verdict,
                    core_support=entry.core_support,
                    core_ready_scope=entry.core_ready_scope,
                    core_blockers=entry.core_blockers,
                )
            )
    return BucketScheduleReport(
        schema=BUCKET_SCHEDULE_SCHEMA,
        consumer_status=BUCKET_CONSUMER_STATUS,
        model=sweep.model,
        chip=sweep.chip,
        default_layout=sweep.parallelism.label(),
        vmem_budget_mib=sweep.vmem_budget_mib,
        max_vmem_tokens=sweep.max_vmem_tokens,
        ridge_tokens=sweep.ridge_tokens,
        buckets=tuple(entries),
    )


def write_bucket_schedule(sweep: AdvisorSweepReport, path: str | Path) -> Path:
    """Write an `advisor.bucket_schedule.v1` JSON schedule to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(build_bucket_schedule(sweep).to_dict(), indent=2) + "\n"
    output_path.write_text(payload_json, encoding="utf-8")
    return output_path
