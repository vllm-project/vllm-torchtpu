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
"""Self-contained offline Layer-1 HTML report generator for TPU roofline plans."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from .chips import CHIPS, ChipSpec, chip_catalog, format_constant_display
from .constants import (
    FRONTIER_2D_TIE_TOLERANCE,
    PERCENT_SCALE,
    PREFIX_RANGE_PAIR_LEN,
    SLO_VERDICT_ESTIMATE_MEETS,
    SLO_VERDICT_RULED_OUT,
    SLO_VERDICT_UNDECIDED,
    UNIT_GIGA,
    UNIT_MILLIS_PER_SECOND,
    UNIT_PETA,
    UNIT_TERA,
)
from .schema import (
    count_hosts,
    format_host_count,
    format_parallelism,
    format_submodule_parallelism,
    pool_label,
    pool_slice_summary,
)

_AXIS_HEADROOM_SCALE = 1.08
_BAR_LABEL_HEADROOM_SCALE = 1.16
_AXIS_TICKS = 4
# Tick-step ladder shared by the static SVG and the inline script.
_AXIS_NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)
_MAX_TICK_DECIMALS = 12
# Decimals each explorer value is embedded and shown with. A cell prints the
# value its row sorts by, so every number is rounded once.
_TPOT_MS_DIGITS = 3
_TTFT_MS_DIGITS = 2
_EST_TPOT_MS_DIGITS = 2
_RATE_DIGITS = 1
_HBM_GB_DIGITS = 1
_FIXED_MAX_VALUE = 1e12
_TICK_STEP_REL_TOL = 1e-9
_MAX_EXPLORER_ROWS_PER_CONC = 80
# Rows kept per sort key and concurrency, so the first rows of every sort are exact.
_EXPLORER_ROWS_PER_SORT_KEY = 15
_DEFAULT_VISIBLE_ROWS = _EXPLORER_ROWS_PER_SORT_KEY
_MAX_FAMILY_ROWS_PER_CONC = 40
_MAX_FAMILY_SAMPLE_LABELS = 2
# Pools kept per concurrency on each pool tab, lowest floor first.
_POOL_ROWS_PER_CONC = _DEFAULT_VISIBLE_ROWS
_BAR_ZERO_CLAMP_MS = 5e-4
_LABEL_BOX_HEIGHT = 18.0
_LABEL_CHAR_WIDTH = 6.6
_LABEL_PAD_X = 10.0
_LABEL_MARGIN = 4.0
_MAX_CALLOUT_ANCHORS = 4
_MIN_ENVELOPE_POINTS = 2
# Sort value of a layout without a TPOT bound, so it never takes a callout.
_NO_BOUND_SORT_MS = 1e9
_NO_LAYOUTS_TEXT = "No physically legal candidates"


def _as_mapping(value: object) -> Mapping[str, object]:
    """Return mapping if value is a Mapping, else empty dict."""
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> list[object]:
    """Return list copy of non-string sequence, else empty list."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce numeric scalar to float, returning default otherwise."""
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_opt_float(value: object) -> float | None:
    """Coerce numeric scalar to finite float, returning None when absent."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    return None


def _as_int(value: object, default: int = 0) -> int:
    """Coerce numeric scalar to int, returning default otherwise."""
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _escape(val: object) -> str:
    """Escape text safely for HTML and inline SVG attributes/nodes."""
    return html.escape(str(val), quote=True)


def _interactivity_val(point: Mapping[str, object]) -> float:
    """Extract est_interactivity_tok_s from a serialized point."""
    return _as_float(point.get("est_interactivity_tok_s"))


def _ttft_val(point: Mapping[str, object]) -> float:
    """Extract est_ttft_ms from a serialized point."""
    return _as_float(point.get("est_ttft_ms"))


def _tpot_val(point: Mapping[str, object]) -> float:
    """Extract est_tpot_ms from a serialized point."""
    return _as_float(point.get("est_tpot_ms"))


def _point_tpot_bound_ms(
    point: Mapping[str, object],
    *,
    accept_len: float = 1.0,
) -> float:
    """Extract physical TPOT lower bound (ms), falling back to decode_floor."""
    bound_raw = _as_opt_float(point.get("bound_tpot_ms"))
    if bound_raw is not None and bound_raw > 0.0:
        return bound_raw
    decode_floor = _as_mapping(point.get("decode_floor"))
    hard_ms = _as_float(decode_floor.get("hard_floor_ms"))
    if hard_ms > 0.0:
        return hard_ms / max(1.0, accept_len)
    return 0.0


def _point_ttft_bound_ms(point: Mapping[str, object]) -> float:
    """Extract physical TTFT lower bound (ms), falling back to prefill_floor."""
    bound_raw = _as_opt_float(point.get("bound_ttft_ms"))
    if bound_raw is not None and bound_raw > 0.0:
        return bound_raw
    prefill_floor = _as_mapping(point.get("prefill_floor"))
    return max(0.0, _as_float(prefill_floor.get("hard_floor_ms")))


def _point_floor_tput(point: Mapping[str, object]) -> float | None:
    """Extract physical throughput ceiling (tok/s/chip) when present."""
    val = _as_opt_float(point.get("floor_throughput_per_chip"))
    return val if val is not None and val > 0.0 else None


def _point_verdict(point: Mapping[str, object]) -> str:
    """Return canonical 3-valued SLO verdict string for a serialized point."""
    raw = point.get("slo_verdict")
    if isinstance(raw, str) and raw in (
        SLO_VERDICT_ESTIMATE_MEETS,
        SLO_VERDICT_UNDECIDED,
        SLO_VERDICT_RULED_OUT,
    ):
        return raw
    return (
        SLO_VERDICT_ESTIMATE_MEETS
        if bool(point.get("slo_met", True))
        else SLO_VERDICT_RULED_OUT
    )


def _point_max_hbm_gb(point: Mapping[str, object]) -> float:
    """Return max HBM used (GB/core) across decode and optional prefill pool."""
    dec_hbm = _as_float(_as_mapping(point.get("decode_pool")).get("hbm_used_gb"))
    pref_hbm = _as_float(_as_mapping(point.get("prefill_pool")).get("hbm_used_gb"))
    return max(dec_hbm, pref_hbm)


def _verdict_badge_html(verdict: str, *, has_slo: bool = True) -> str:
    """Render pill badge for the 3-valued SLO verdict, or No SLO when none is set."""
    if not has_slo:
        return '<span class="v-badge v-none">No SLO</span>'
    if verdict == SLO_VERDICT_ESTIMATE_MEETS:
        return '<span class="v-badge v-meets">Est. Meets</span>'
    if verdict == SLO_VERDICT_UNDECIDED:
        return '<span class="v-badge v-undecided">Undecided</span>'
    return '<span class="v-badge v-ruled">Ruled Out</span>'


def _nice_axis_max(raw_max: float) -> float:
    """Round the axis upper bound up to a tick boundary at or above the data."""
    if not math.isfinite(raw_max) or raw_max <= 0.0:
        return 1.0
    step = raw_max * _AXIS_HEADROOM_SCALE / float(_AXIS_TICKS)
    scale = 10.0 ** math.floor(math.log10(step))
    frac = step / scale
    nice_step = _AXIS_NICE_STEPS[-1]
    for candidate in _AXIS_NICE_STEPS:
        if frac <= candidate:
            nice_step = candidate
            break
    return nice_step * scale * float(_AXIS_TICKS)


def _fixed(value: float, digits: int, *, grouped: bool = False) -> str:
    """Format half-up on the shortest decimal form, matching the inline script.

    Every formatted quantity is non-negative, so values clamp to [0, 1e12].
    Table cells use at most 3 decimals; more decimals occur only on axis ticks
    of small values.
    """
    clamped = min(float(value), _FIXED_MAX_VALUE) if value > 0.0 else 0.0
    quantum = Decimal(1).scaleb(-digits)
    rounded = Decimal(repr(clamped)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:,f}" if grouped else f"{rounded:f}"


def _to_fixed(value: float, digits: int) -> str:
    """Format like JavaScript Number.toFixed: exact binary value, ties round up."""
    quantum = Decimal(1).scaleb(-digits)
    return f"{Decimal(float(value)).quantize(quantum, rounding=ROUND_HALF_UP):f}"


def _tick_decimals(step: float) -> int:
    """Return the fewest decimals (at most 12) that print a tick step exactly."""
    for digits in range(_MAX_TICK_DECIMALS):
        if abs(round(step, digits) - step) <= _TICK_STEP_REL_TOL * abs(step):
            return digits
    return _MAX_TICK_DECIMALS


def _format_axis_tick(tick_val: float, step: float) -> str:
    """Format an SVG axis tick with the decimals its tick step needs."""
    if tick_val <= 0.0:
        return "0"
    return _fixed(tick_val, _tick_decimals(step), grouped=True)


def _shown_of_total(shown: int, total: int) -> str:
    """Format a row count as 'shown' or 'shown of total'."""
    return f"{shown:,}" if shown >= total else f"{shown:,} of {total:,}"


def _format_chip_count(chips: int) -> str:
    """Format chip count with singular/plural grammar."""
    return f"{chips} chip" if chips == 1 else f"{chips} chips"


def _format_dtype_bytes(byte_value: object) -> str:
    """Format a serialized dtype width in bytes for report headers."""
    return f"{_as_float(byte_value):.6g}B"


def _format_opt_ms(value: object, digits: int = 2) -> str:
    """Format optional latency (ms) value or '-' when absent."""
    val = _as_opt_float(value)
    if val is None or val <= 0.0:
        return "-"
    return _fixed(val, digits)


def _format_opt_rate(value: object) -> str:
    """Format optional throughput (tok/s/chip) value or '-' when absent."""
    val = _as_opt_float(value)
    if val is None or val <= 0.0:
        return "-"
    return _fixed(val, 1, grouped=True)


def _detect_has_moe(
    payload: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> bool:
    """Detect whether the reported model has routed MoE layers."""
    if "has_moe" in payload:
        return bool(payload["has_moe"])
    for cand in candidates:
        for key in ("decode_pool", "prefill_pool"):
            pool = _as_mapping(cand.get(key))
            if _as_int(pool.get("ep"), 1) > 1 or str(pool.get("moe", "")) == "ep":
                return True
    return False


def _pool_inline_label(
    pool: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> str:
    """Format compact single-line label for one pool mesh plan."""
    if not pool:
        return "-"
    if not has_moe:
        return pool_label(pool)
    summary = pool_slice_summary(pool, chip=chip)
    submod = format_submodule_parallelism(pool, has_moe=True)
    return f"{summary} ({submod})"


def _pool_cell_parts(
    pool: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> tuple[str, str]:
    """Return (main_text, sub_text) describing a pool's slice and parallelism."""
    if not pool:
        return ("-", "")
    sub = pool_slice_summary(pool, chip=chip)
    main = (
        format_submodule_parallelism(pool, has_moe=True)
        if has_moe
        else format_parallelism(pool)
    )
    return (main, sub)


def _pools_display(
    point: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> str:
    """Build single-line layout summary for headers, tooltips, and charts."""
    pref = _as_mapping(point.get("prefill_pool"))
    dec = _as_mapping(point.get("decode_pool"))
    if pref and dec:
        return (
            f"P: {_pool_inline_label(pref, has_moe=has_moe, chip=chip)} / "
            f"D: {_pool_inline_label(dec, has_moe=has_moe, chip=chip)}"
        )
    if dec:
        return _pool_inline_label(dec, has_moe=has_moe, chip=chip)
    return "-"


def _point_dedup_key(point: Mapping[str, object]) -> tuple[object, ...]:
    """Build structural + concurrency deduplication key for a serialized point."""
    dec = _as_mapping(point.get("decode_pool"))
    pref = _as_mapping(point.get("prefill_pool"))

    def _p_sig(pool: Mapping[str, object]) -> tuple[object, ...]:
        if not pool:
            return ()
        return (
            str(pool.get("topology", "")),
            _as_int(pool.get("instances"), 1),
            _as_int(pool.get("tp"), 1),
            _as_int(pool.get("dp"), 1),
            _as_int(pool.get("ep"), 1),
            _as_int(pool.get("pp"), 1),
            _as_int(pool.get("pcp"), 1),
            _as_int(pool.get("kv_split"), 1),
            str(pool.get("moe", "")),
            str(pool.get("moe_dispatch", "")),
            str(pool.get("residual", "")),
        )

    return (
        str(point.get("mode", "")),
        _as_int(point.get("concurrency")),
        _as_int(point.get("total_chips")),
        _p_sig(pref),
        _p_sig(dec),
    )


def _collect_all_points(
    payload: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], tuple[object, ...] | None]:
    """Collect deduplicated points across candidates, frontier, top_k, and families."""
    seen: dict[tuple[object, ...], Mapping[str, object]] = {}

    def _add(item: object) -> None:
        pt = _as_mapping(item)
        if not pt or not _as_mapping(pt.get("decode_pool")):
            return
        key = _point_dedup_key(pt)
        if key not in seen:
            seen[key] = pt

    for cand in _as_sequence(payload.get("candidates")):
        _add(cand)
    for pt in _as_sequence(payload.get("frontier")):
        _add(pt)
    for entry in _as_sequence(payload.get("top_k")):
        _add(_as_mapping(entry).get("point"))
    rec = _as_mapping(payload.get("recommended"))
    if rec:
        _add(rec)
    ref_report = _as_mapping(payload.get("reference"))
    ref_key: tuple[object, ...] | None = None
    if ref_report and bool(ref_report.get("found")):
        ref_pt = _as_mapping(ref_report.get("point"))
        if ref_pt and _as_mapping(ref_pt.get("decode_pool")):
            _add(ref_pt)
            ref_key = _point_dedup_key(ref_pt)
    raw_conc_top = _as_mapping(payload.get("concurrency_top_k"))
    for conc_entries in raw_conc_top.values():
        for entry in _as_sequence(conc_entries):
            _add(_as_mapping(entry).get("point"))
    raw_fams = _as_mapping(payload.get("bound_families_by_concurrency"))
    for fam_list in raw_fams.values():
        for fam in _as_sequence(fam_list):
            members = _as_sequence(_as_mapping(fam).get("members"))
            if members:
                _add(members[0])
    return list(seen.values()), ref_key


def _envelope_2d_upper(
    points: Sequence[tuple[float, float, int]],
) -> list[tuple[float, float, int]]:
    """Compute monotonic 2D upper-right envelope over (x, y, idx) tuples."""
    if not points:
        return []
    ordered = sorted(points, key=lambda item: (-item[0], -item[1], item[2]))
    env: list[tuple[float, float, int]] = []
    best_y = -1.0
    for x_val, y_val, idx in ordered:
        if y_val > best_y + FRONTIER_2D_TIE_TOLERANCE:
            env.append((x_val, y_val, idx))
            best_y = y_val
    env.sort(key=lambda item: item[0])
    return env


def _place_collision_free_labels(
    anchors: Sequence[tuple[float, float, str, str]],
    plot_bounds: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float, float, str, str]]:
    """Place up to 4 callout labels with bounding-box overlap rejection.

    Returns list of ``(px, py, box_x, box_y, box_w, label, color)``.
    """
    min_x, max_x, min_y, max_y = plot_bounds
    placed_rects: list[tuple[float, float, float, float]] = []
    out: list[tuple[float, float, float, float, float, str, str]] = []
    offsets = (
        (12.0, -24.0),
        (-12.0, -24.0),
        (12.0, 10.0),
        (-12.0, 10.0),
        (0.0, -34.0),
        (0.0, 22.0),
        (24.0, -38.0),
        (-24.0, -38.0),
        (24.0, 28.0),
        (-24.0, 28.0),
    )

    def _overlaps(
        rect: tuple[float, float, float, float],
    ) -> bool:
        rx0, ry0, rx1, ry1 = rect
        for ox0, oy0, ox1, oy1 in placed_rects:
            if (
                rx0 < ox1 + _LABEL_MARGIN
                and rx1 + _LABEL_MARGIN > ox0
                and ry0 < oy1 + _LABEL_MARGIN
                and ry1 + _LABEL_MARGIN > oy0
            ):
                return True
        return False

    for px, py, text, color in anchors[:_MAX_CALLOUT_ANCHORS]:
        box_w = min(210.0, len(text) * _LABEL_CHAR_WIDTH + _LABEL_PAD_X)
        box_h = _LABEL_BOX_HEIGHT
        chosen: tuple[float, float] | None = None
        for dx, dy in offsets:
            cand_x = px + dx if dx >= 0.0 else px + dx - box_w
            cand_y = py + dy
            cand_x = max(min_x + 4.0, min(max_x - box_w - 4.0, cand_x))
            cand_y = max(min_y + 4.0, min(max_y - box_h - 4.0, cand_y))
            rect = (cand_x, cand_y, cand_x + box_w, cand_y + box_h)
            if not _overlaps(rect):
                chosen = (cand_x, cand_y)
                placed_rects.append(rect)
                break
        if chosen is not None:
            out.append((px, py, chosen[0], chosen[1], box_w, text, color))
    return out


def _explorer_row_label(row: Mapping[str, object]) -> str:
    """Return the one-line layout label the inline script shows for a row."""
    prefill, decode = str(row.get("pm") or ""), str(row.get("dm") or "")
    return f"P: {prefill} / D: {decode}" if prefill else decode


def _row_plot_xy(row: Mapping[str, object], mode: str) -> tuple[float, float]:
    """Return a row's trade-off chart coordinates for the given plot mode."""
    if mode == "bound":
        bound_ms = _as_float(row.get("bt"))
        x_val = UNIT_MILLIS_PER_SECOND / bound_ms if bound_ms > 0.0 else 0.0
        ceiling = _as_opt_float(row.get("fc"))
        y_val = ceiling if ceiling is not None else _as_float(row.get("et"))
    else:
        x_val, y_val = _as_float(row.get("ei")), _as_float(row.get("et"))
    return max(0.0, x_val), max(0.0, y_val)


def _tradeoff_point_title(row: Mapping[str, object], *, has_slo: bool) -> str:
    """Return the hover text of one plotted layout."""
    verdict_txt = str(row.get("v") or "") if has_slo else "no SLO"
    return (
        f"{_explorer_row_label(row)} | c={_as_int(row.get('c'))} | {verdict_txt} | "
        f"TPOT bound={_format_opt_ms(row.get('bt'), _TPOT_MS_DIGITS)} ms | "
        f"Ceiling={_format_opt_rate(row.get('fc'))} tok/s/chip | "
        f"Est={_format_opt_rate(row.get('et'))} tok/s/chip, "
        f"{_fixed(_as_float(row.get('ei')), _RATE_DIGITS)} tok/s"
    )


def _render_tradeoff_svg(
    rows: Sequence[tuple[int, Mapping[str, object]]],
    *,
    selected_idx: int,
    mode: str = "est",
    has_slo: bool = True,
) -> str:
    """Render the trade-off chart of explorer rows as the inline script redraws it.

    ``rows`` pairs each row with its index in the embedded row list, in display
    order. Geometry, styles, callouts, and hover text follow ``renderPlot``.
    """
    width, height = 920, 420
    pad_left, pad_right, pad_top, pad_bottom = 86, 36, 26, 54
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    if not rows:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'fill="#64748b" font-size="14">{_NO_LAYOUTS_TEXT}</text></svg>'
        )

    coords = [_row_plot_xy(row, mode) for _, row in rows]
    max_x = _nice_axis_max(max(x_val for x_val, _ in coords))
    max_y = _nice_axis_max(max(y_val for _, y_val in coords))

    def _sx(x_val: float) -> float:
        return pad_left + (x_val / max_x) * plot_w

    def _sy(y_val: float) -> float:
        return pad_top + plot_h - (y_val / max_y) * plot_h

    x_title = (
        "Max Bound Interactivity (1000 / bound_tpot_ms, tok/s)"
        if mode == "bound"
        else "Est. Interactivity (tok/s)"
    )
    y_title = (
        "Physical Throughput Ceiling (tok/s/chip)"
        if mode == "bound"
        else "Est. Token Throughput per Chip (tok/s/chip)"
    )

    svg: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" '
        'fill="#f8fafc" stroke="#e2e8f0" rx="4"/>',
    ]
    x_step = max_x / float(_AXIS_TICKS)
    y_step = max_y / float(_AXIS_TICKS)
    for tick_idx in range(_AXIS_TICKS + 1):
        frac = tick_idx / float(_AXIS_TICKS)
        x_tick = _format_axis_tick(frac * max_x, x_step)
        y_tick = _format_axis_tick(frac * max_y, y_step)
        gx = _to_fixed(pad_left + frac * plot_w, 1)
        gy = pad_top + plot_h - frac * plot_h
        svg.append(
            f'<line x1="{gx}" y1="{pad_top}" x2="{gx}" '
            f'y2="{pad_top + plot_h}" stroke="#e2e8f0" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'<line x1="{pad_left}" y1="{_to_fixed(gy, 1)}" x2="{pad_left + plot_w}" '
            f'y2="{_to_fixed(gy, 1)}" stroke="#e2e8f0" stroke-dasharray="3,3"/>'
        )
        svg.append(
            f'<text x="{gx}" y="{height - 28}" text-anchor="middle" '
            f'font-size="11.5" fill="#64748b">{x_tick}</text>'
        )
        svg.append(
            f'<text x="{pad_left - 10}" y="{_to_fixed(gy + 4, 1)}" text-anchor="end" '
            f'font-size="11.5" fill="#64748b">{y_tick}</text>'
        )
    y_mid = _to_fixed(pad_top + plot_h / 2, 1)
    svg.append(
        f'<text x="{_to_fixed(pad_left + plot_w / 2, 1)}" y="{height - 8}" '
        'text-anchor="middle" font-size="12.5" font-weight="600" fill="#334155">'
        f"{_escape(x_title)}</text>"
    )
    svg.append(
        f'<text x="20" y="{y_mid}" text-anchor="middle" '
        f'font-size="12.5" font-weight="600" fill="#334155" '
        f'transform="rotate(-90 20 {y_mid})">{_escape(y_title)}</text>'
    )

    # The dashed envelope and the callouts skip ruled-out layouts when any survive.
    surviving = [
        pos
        for pos, (_, row) in enumerate(rows)
        if row.get("v") != SLO_VERDICT_RULED_OUT
    ]
    envelope = _envelope_2d_upper([(*coords[pos], pos) for pos in surviving])
    if len(envelope) >= _MIN_ENVELOPE_POINTS:
        pts_str = " ".join(
            f"{_to_fixed(_sx(x_val), 1)},{_to_fixed(_sy(y_val), 1)}"
            for x_val, y_val, _ in envelope
        )
        env_stroke = "#0284c7" if mode == "bound" else "#2563eb"
        svg.append(
            f'<polyline points="{pts_str}" fill="none" stroke="{env_stroke}" '
            'stroke-width="2" stroke-dasharray="6,4" opacity="0.8"/>'
        )

    for (row_idx, row), (x_val, y_val) in zip(rows, coords, strict=True):
        verdict = row.get("v")
        if verdict == SLO_VERDICT_RULED_OUT:
            fill, stroke, opacity = "#94a3b8", "#64748b", "0.42"
        elif verdict == SLO_VERDICT_UNDECIDED:
            fill, stroke, opacity = "#fef3c7", "#d97706", "0.85"
        else:
            fill, stroke, opacity = "#2563eb", "#1d4ed8", "0.85"
        selected = row_idx == selected_idx
        svg.append(
            f'<circle cx="{_to_fixed(_sx(x_val), 1)}" cy="{_to_fixed(_sy(y_val), 1)}" '
            f'r="{"6.5" if selected else "4.5"}" fill="{fill}" '
            f'stroke="{"#0f172a" if selected else stroke}" '
            f'stroke-width="{"2" if selected else "1.2"}" opacity="{opacity}" '
            'style="cursor:pointer">'
            f"<title>{_escape(_tradeoff_point_title(row, has_slo=has_slo))}</title>"
            "</circle>"
        )

    # Mode-matched callouts; a layout that holds two roles keeps its first label.
    pool = surviving or list(range(len(rows)))
    callouts: dict[int, tuple[str, str, bool]] = {}
    reference_pos = next(
        (pos for pos, (_, row) in enumerate(rows) if bool(row.get("ref"))), None
    )
    if reference_pos is not None:
        callouts[reference_pos] = ("Reference", "#7c3aed", True)
    if mode == "bound":
        min_bound_pos = min(
            pool,
            key=lambda pos: _as_float(rows[pos][1].get("bt")) or _NO_BOUND_SORT_MS,
        )
        callouts.setdefault(min_bound_pos, ("Min Bound TPOT", "#059669", False))
        if any(rows[pos][1].get("fc") is not None for pos in pool):
            max_ceiling_pos = max(
                pool, key=lambda pos: _as_float(rows[pos][1].get("fc"))
            )
            callouts.setdefault(max_ceiling_pos, ("Max Ceiling Tput", "#0284c7", False))
    else:
        max_estimate_pos = max(pool, key=lambda pos: _as_float(rows[pos][1].get("et")))
        callouts.setdefault(max_estimate_pos, ("Max Est. Tput", "#d97706", False))

    anchor_inputs: list[tuple[float, float, str, str]] = []
    for pos, (tag, color, dashed) in callouts.items():
        px, py = _sx(coords[pos][0]), _sy(coords[pos][1])
        dash = ' stroke-dasharray="3,2"' if dashed else ""
        svg.append(
            f'<circle cx="{_to_fixed(px, 1)}" cy="{_to_fixed(py, 1)}" r="9.5" '
            f'fill="none" stroke="{color}" stroke-width="2.2"{dash}/>'
        )
        anchor_inputs.append((px, py, tag, color))

    placed = _place_collision_free_labels(
        anchor_inputs,
        (
            float(pad_left),
            float(pad_left + plot_w),
            float(pad_top),
            float(pad_top + plot_h),
        ),
    )
    for px, py, bx, by, bw, label, color in placed:
        anchor_x = bx + bw / 2.0
        anchor_y = by + _LABEL_BOX_HEIGHT / 2.0
        svg.append(
            f'<line x1="{_to_fixed(px, 1)}" y1="{_to_fixed(py, 1)}" '
            f'x2="{_to_fixed(anchor_x, 1)}" y2="{_to_fixed(anchor_y, 1)}" '
            f'stroke="{color}" stroke-width="1" opacity="0.65"/>'
        )
        svg.append(
            f'<rect x="{_to_fixed(bx, 1)}" y="{_to_fixed(by, 1)}" '
            f'width="{_to_fixed(bw, 1)}" height="{_LABEL_BOX_HEIGHT:.0f}" rx="4" '
            f'fill="#ffffff" fill-opacity="0.94" stroke="{color}" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{_to_fixed(bx + 5.0, 1)}" y="{_to_fixed(by + 12.5, 1)}" '
            f'font-size="11" font-weight="600" fill="{color}">{_escape(label)}</text>'
        )

    svg.append("</svg>")
    return "".join(svg)


def _render_limits_svg(floor: Mapping[str, object] | None) -> str:
    """Render decode-step bar chart splitting physical bounds from est overhead."""
    width, height = 920, 345
    pad_left, pad_right, pad_top = 175, 115, 28

    if not floor:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" '
            'text-anchor="middle" fill="#64748b" font-size="14">'
            "No decode floor data available</text></svg>"
        )

    hbm_ms = _as_float(floor.get("hbm_bound_ms", floor.get("hbm_ms")))
    matmul_ms = _as_float(floor.get("matmul_bound_ms", floor.get("matmul_ms")))
    vector_ms = _as_float(floor.get("vector_bound_ms"))
    ici_ms = _as_float(floor.get("coll_bound_ms", floor.get("ici_ms")))
    hard_floor_ms = _as_float(floor.get("hard_floor_ms"))
    chain_ms = _as_float(floor.get("chain_ms"))
    hop_ms = _as_float(floor.get("hop_latency_ms"))

    bound_items = [
        ("HBM Bound", hbm_ms, "#3b82f6"),
        ("MatMul Bound", matmul_ms, "#10b981"),
        ("Vector Bound", vector_ms, "#06b6d4"),
        ("Collective Bound", ici_ms, "#f59e0b"),
        ("Hard Floor", hard_floor_ms, "#ef4444"),
    ]
    est_items = [
        ("Serial Op Overhead", chain_ms, "#8b5cf6"),
        ("Hop Latency", hop_ms, "#ec4899"),
    ]
    all_vals = [
        val
        for _, val, _ in (*bound_items, *est_items)
        if math.isfinite(val) and val > 0.0
    ]
    max_val = (max(all_vals, default=1.0) or 1.0) * _BAR_LABEL_HEADROOM_SCALE
    plot_w = width - pad_left - pad_right
    bar_h, bar_gap = 22, 12

    svg: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{pad_left}" y="16" font-size="11" font-weight="700" '
        'fill="#0f172a">PHYSICAL LOWER BOUNDS [bound_*]</text>',
    ]

    def _draw_row(y_pos: int, label: str, val: float, color: str) -> None:
        if not math.isfinite(val) or val < _BAR_ZERO_CLAMP_MS:
            bar_w = 0.0
            width_attr = "0"
        else:
            bar_w = max(2.5, (val / max_val) * plot_w)
            width_attr = f"{bar_w:.1f}"
        weight = "700" if label == "Hard Floor" else "500"
        svg.extend(
            [
                f'<text x="{pad_left - 14}" y="{y_pos + 15}" text-anchor="end" '
                f'font-size="12.5" font-weight="{weight}" fill="#1e293b">'
                f"{_escape(label)}</text>",
                f'<rect x="{pad_left}" y="{y_pos}" width="{width_attr}" '
                f'height="{bar_h}" fill="{color}" rx="4">'
                f"<title>{_escape(label)}: {val:.3f} ms</title></rect>",
                f'<text x="{pad_left + bar_w + 10:.1f}" y="{y_pos + 15}" '
                f'font-size="12" font-weight="{weight}" fill="#334155">'
                f"{val:.3f} ms</text>",
            ]
        )

    cur_y = pad_top
    for label, val, color in bound_items:
        _draw_row(cur_y, label, val, color)
        cur_y += bar_h + bar_gap

    div_y = cur_y + 6
    svg.append(
        f'<line x1="{pad_left}" y1="{div_y}" x2="{width - 24}" y2="{div_y}" '
        'stroke="#cbd5e1" stroke-dasharray="4,4"/>'
    )
    svg.append(
        f'<text x="{pad_left}" y="{div_y + 18}" font-size="11" font-weight="700" '
        'fill="#6b21a8">SERIAL ESTIMATE OVERHEAD [est_*]</text>'
    )
    cur_y = div_y + 28
    for label, val, color in est_items:
        _draw_row(cur_y, label, val, color)
        cur_y += bar_h + bar_gap

    svg.append("</svg>")
    return "".join(svg)


def _render_memory_bar(
    pool: Mapping[str, object],
    y: int,
    plot_cfg: tuple[int, int, int],
    default_limit_gb: float,
) -> list[str]:
    """Render one stacked horizontal HBM bar row for a pool."""
    pad_left, plot_w, bar_h = plot_cfg
    role = str(pool.get("role") or "pool")
    topology = str(pool.get("topology") or "")

    # The inline script redraws this bar from values rounded to 1 decimal.
    def _gb(value: float) -> float:
        return round(max(0.0, value), _HBM_GB_DIGITS)

    limit_gb = _gb(_as_float(pool.get("hbm_limit_gb"), default_limit_gb))
    if limit_gb <= 0.0:
        limit_gb = _gb(default_limit_gb)
    weights_gb = _gb(_as_float(pool.get("hbm_weights_gb")))
    kv_cache_gb = _gb(_as_float(pool.get("hbm_kv_gb")))
    state_gb = _gb(_as_float(pool.get("hbm_state_gb")))
    used_gb = _gb(
        _as_float(pool.get("hbm_used_gb"), weights_gb + kv_cache_gb + state_gb)
    )
    free_gb = max(0.0, limit_gb - used_gb)

    bar_title = _escape(f"{role.capitalize()} ({topology})")
    parts = [
        f'<text x="{pad_left - 14}" y="{y + 18}" text-anchor="end" '
        f'font-size="13" font-weight="500" fill="#1e293b">{bar_title}</text>'
    ]
    segments = [
        (weights_gb, "#3b82f6", f"Weights: {_fixed(weights_gb, _HBM_GB_DIGITS)} GB"),
        (kv_cache_gb, "#10b981", f"KV Cache: {_fixed(kv_cache_gb, _HBM_GB_DIGITS)} GB"),
        (
            state_gb,
            "#f59e0b",
            f"Recurrent State: {_fixed(state_gb, _HBM_GB_DIGITS)} GB",
        ),
        (free_gb, "#e2e8f0", f"Headroom: {_fixed(free_gb, _HBM_GB_DIGITS)} GB"),
    ]
    cur_x = float(pad_left)
    max_x_px = float(pad_left + plot_w)
    if limit_gb > 0.0:
        for gb_val, fill, title in segments:
            remaining_w = max(0.0, max_x_px - cur_x)
            if remaining_w <= 0.0 or gb_val <= 0.0:
                continue
            seg_w = (
                remaining_w
                if fill == "#e2e8f0"
                else min(remaining_w, (gb_val / limit_gb) * plot_w)
            )
            if seg_w <= 0.0:
                continue
            parts.append(
                f'<rect x="{cur_x:.1f}" y="{y}" width="{seg_w:.1f}" '
                f'height="{bar_h}" fill="{fill}">'
                f"<title>{_escape(title)}</title></rect>"
            )
            cur_x += seg_w

    util_pct = (
        math.floor((used_gb / limit_gb) * PERCENT_SCALE + 0.5) if limit_gb > 0.0 else 0
    )
    usage_text = (
        f"{_fixed(used_gb, _HBM_GB_DIGITS)} / {_fixed(limit_gb, _HBM_GB_DIGITS)} GB"
    )
    parts.extend(
        [
            f'<rect x="{pad_left}" y="{y}" width="{plot_w}" height="{bar_h}" '
            'fill="none" stroke="#94a3b8" rx="4"/>',
            f'<text x="{pad_left + plot_w + 12}" y="{y + 18}" font-size="12.5" '
            f'font-weight="600" fill="#334155">{usage_text} '
            f"({util_pct}%)</text>",
        ]
    )
    return parts


def _render_memory_svg(
    pools: Sequence[Mapping[str, object]],
    default_limit_gb: float,
) -> str:
    """Render inline stacked horizontal bar chart for HBM footprint vs capacity."""
    width, row_h = 920, 62
    pad_left, pad_right, pad_top, bar_h = 175, 185, 22, 26
    height = pad_top + max(1, len(pools)) * row_h + 8
    plot_w = width - pad_left - pad_right

    if not pools:
        return (
            f'<svg viewBox="0 0 {width} 120" class="chart-svg" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="60" text-anchor="middle" '
            'fill="#64748b" font-size="14">'
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
    svg_parts.append("</svg>")
    return "".join(svg_parts)


_MEMORY_LEGEND_HTML = (
    '<div id="memory-legend" class="chart-legend">'
    '<span class="legend-swatch" style="background:#3b82f6;"></span>Weights'
    '<span class="legend-swatch" style="background:#10b981;"></span>KV Cache'
    '<span class="legend-swatch" style="background:#f59e0b;"></span>Recurrent State'
    '<span class="legend-swatch" style="background:#e2e8f0;'
    'border:1px solid #94a3b8;"></span>Free Headroom'
    "</div>"
)


def _derive_default_hbm_limit_gb(payload: Mapping[str, object]) -> float:
    """Derive per-core HBM limit (GB) from chip_spec and hbm_reserve_gb."""
    chip_spec = _as_mapping(payload.get("chip_spec"))
    reserve_gb = max(0.0, _as_float(payload.get("hbm_reserve_gb"), 0.0))
    fallback_chip = CHIPS["tpu7x"]
    hbm_bytes = _as_float(chip_spec.get("hbm_bytes"), float(fallback_chip.hbm_bytes))
    return max(0.0, hbm_bytes / UNIT_GIGA - reserve_gb)


def _build_dynamic_chip_spec(
    chip_spec: Mapping[str, object],
    fallback: ChipSpec,
    chip_name: str,
) -> ChipSpec:
    """Build ChipSpec dynamically from serialized payload chip_spec mapping."""
    kwargs: dict[str, object] = {}
    for spec_field in fields(ChipSpec):
        field_name = spec_field.name
        if field_name == "name":
            kwargs[field_name] = chip_name
            continue
        default_val = getattr(fallback, field_name)
        raw_val = chip_spec.get(field_name)
        if isinstance(default_val, bool):
            kwargs[field_name] = bool(raw_val) if raw_val is not None else default_val
        elif isinstance(default_val, int):
            kwargs[field_name] = (
                int(_as_float(raw_val, float(default_val)))
                if raw_val is not None
                else default_val
            )
        elif isinstance(default_val, float):
            kwargs[field_name] = _as_float(raw_val, default_val)
        else:
            kwargs[field_name] = (
                str(raw_val) if raw_val is not None and str(raw_val) else default_val
            )
    return ChipSpec(**kwargs)  # type: ignore[arg-type]


def _payload_chip_spec(payload: Mapping[str, object]) -> ChipSpec | None:
    """Reconstruct ChipSpec from payload['chip_spec'] when present."""
    chip_spec = _as_mapping(payload.get("chip_spec"))
    if not chip_spec:
        return None
    fallback = CHIPS["tpu7x"]
    chip_name = str(chip_spec.get("name") or payload.get("chip") or fallback.name)
    return _build_dynamic_chip_spec(chip_spec, fallback, chip_name)


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
    chip_name = str(chip_spec.get("name") or payload.get("chip") or fallback.name)
    hbm_gb = (
        _as_float(chip_spec.get("hbm_bytes"), float(fallback.hbm_bytes)) / UNIT_GIGA
    )
    reserve_gb = max(0.0, _as_float(payload.get("hbm_reserve_gb"), 0.0))
    usable_gb = max(0.0, hbm_gb - reserve_gb)
    reserve_note = (
        f" minus <code>{reserve_gb:.2f} GB</code> HBM reserve"
        if reserve_gb > 0.0
        else ""
    )
    hbm_tbs = _as_float(chip_spec.get("hbm_bw"), fallback.hbm_bw) / UNIT_TERA
    bf16_pf = _as_float(chip_spec.get("bf16_flops"), fallback.bf16_flops) / UNIT_PETA
    fp8_pf = _as_float(chip_spec.get("fp8_flops"), fallback.fp8_flops) / UNIT_PETA
    extra_parts: list[str] = []
    for key, label, default_val in (
        ("fp4_flops", "FP4", fallback.fp4_flops),
        ("int8_flops", "INT8", fallback.int8_flops),
        ("int4_flops", "INT4", fallback.int4_flops),
    ):
        rate_pf = _as_float(chip_spec.get(key), default_val) / UNIT_PETA
        if rate_pf > 0.0:
            extra_parts.append(f"<code>{rate_pf:.3f} PFLOP/s</code> {label}, ")
    vector_tf = (
        _as_float(chip_spec.get("vector_flops"), fallback.vector_flops) / UNIT_TERA
    )
    if vector_tf > 0.0:
        extra_parts.append(f"<code>{vector_tf:.2f} TFLOP/s</code> Vector, ")
    extra_compute = "".join(extra_parts)
    ici_gbs = _as_float(chip_spec.get("ici_bw"), fallback.ici_bw) / UNIT_GIGA
    d2d_gbs = _as_float(chip_spec.get("d2d_bw"), fallback.d2d_bw) / UNIT_GIGA
    return (
        f"      <li><strong>{_escape(chip_name)} Per-Core Specs:</strong> "
        f"<code>{hbm_gb:.1f} GB</code> HBM capacity "
        f"(<code>{usable_gb:.2f} GB</code> usable{reserve_note}), "
        f"<code>{hbm_tbs:.2f} TB/s</code> HBM BW, "
        f"<code>{bf16_pf:.3f} PFLOP/s</code> BF16, "
        f"<code>{fp8_pf:.3f} PFLOP/s</code> FP8, {extra_compute}"
        f"<code>{ici_gbs:.0f} GB/s</code> bidir ICI, "
        f"<code>{d2d_gbs:.0f} GB/s</code> intra-chip D2D.</li>"
    )


def _format_assumption_entry(name: str, entry: object) -> str:
    """Format one supplied_values or solver_assumptions entry."""
    if isinstance(entry, Mapping):
        raw_name = str(entry.get("name") or name)
        raw_flag = str(entry.get("flag") or "")
        label = raw_name if raw_flag == "--chip-spec" else (raw_flag or raw_name)
        val = entry.get("value")
        is_number = isinstance(val, (int, float)) and not isinstance(val, bool)
        unit = str(entry.get("unit") or "") if is_number else ""
        val_txt = str(val)
        suffix = f" {unit}" if unit else ""
        return f"<code>{_escape(label)}={_escape(val_txt)}</code>{_escape(suffix)}"
    return f"<code>{_escape(name)}={_escape(entry)}</code>"


def _format_supplied_values_html(raw_supplied: object) -> str:
    """Format supplied_values or solver_assumptions mapping/sequence."""
    items: list[str] = []
    if isinstance(raw_supplied, Mapping):
        for key, val in raw_supplied.items():
            items.append(_format_assumption_entry(str(key), val))
    elif isinstance(raw_supplied, Sequence) and not isinstance(
        raw_supplied, (str, bytes)
    ):
        for idx, val in enumerate(raw_supplied):
            entry_name = (
                str(val.get("name", f"item_{idx}"))
                if isinstance(val, Mapping)
                else f"item_{idx}"
            )
            items.append(_format_assumption_entry(entry_name, val))
    return ", ".join(items)


def _render_provenance_item(payload: Mapping[str, object]) -> str:
    """Render Provenance list item when provenance metadata exists."""
    prov = _as_mapping(payload.get("provenance"))
    if not prov:
        return ""
    segments: list[str] = []
    calibrated = prov.get("calibrated")
    if calibrated is not None:
        segments.append(
            f"calibrated=<code>{'true' if bool(calibrated) else 'false'}</code>"
        )
    ranking_basis = prov.get("ranking_basis")
    if isinstance(ranking_basis, str) and ranking_basis:
        segments.append(f"ranking_basis=<code>{_escape(ranking_basis)}</code>")
    supplied_html = _format_supplied_values_html(prov.get("supplied_values"))
    if supplied_html:
        segments.append(f"supplied={supplied_html}")
    if not segments:
        return ""
    return f"      <li><strong>Provenance:</strong> {' &middot; '.join(segments)}.</li>"


def _render_solver_assumptions_items(payload: Mapping[str, object]) -> list[str]:
    """Render guarded Solver Assumptions list item when present in payload."""
    formatted = _format_supplied_values_html(payload.get("solver_assumptions"))
    if not formatted:
        return []
    return [f"      <li><strong>Solver Assumptions:</strong> {formatted}.</li>"]


def _render_glossary_pool_floor_item(payload: Mapping[str, object]) -> str:
    """Render the pool-tab floor glossary entries when the payload has pool tables."""
    if not (payload.get("prefill_pools") or payload.get("decode_pools")):
        return ""
    return (
        "<li><strong>TPOT Floor (Decode Pools tab):</strong> the hard step floor of "
        "one decode pool at the batch that concurrency puts on it. It can be higher "
        "than the TPOT Bound of a pair that uses the pool, because the bound uses "
        "the fewest resident sequences.</li>"
        "<li><strong>TTFT Floor (Prefill Pools tab):</strong> the hard floor of one "
        "prefill pool's time to first token. It is the same at every "
        "concurrency.</li>"
    )


def _render_glossary_unit_item(
    payload: Mapping[str, object],
    chip: ChipSpec | None,
) -> str:
    """Render chip-specific physical unit glossary entry."""
    fallback = chip if chip is not None else CHIPS["tpu7x"]
    chip_name = str(payload.get("chip") or fallback.name)
    cph = fallback.chips_per_host
    cpc = fallback.cores_per_chip
    cores_host = cph * cpc
    chips_label = _format_chip_count(cph)
    core_word = "TensorCore" if cpc == 1 else "TensorCores"
    host_core_word = "TensorCore" if cores_host == 1 else "TensorCores"
    host_fit = _as_mapping(payload.get("single_host_fit"))
    raw_topo = host_fit.get("topology")
    host_topo = (
        str(raw_topo) if isinstance(raw_topo, str) and raw_topo else f"{cph}-chip"
    )
    return (
        f"<li><strong>Physical Units ({_escape(chip_name)}):</strong> "
        f"1 chip = <code>{cpc} {core_word}</code>; "
        f"1 host = <code>{chips_label}</code> "
        f"(<code>{cores_host} {host_core_word}</code>); "
        f"<code>{_escape(host_topo)}</code> = {chips_label} / 1 host. "
        "Each slice prints its topology, chip count, and host count.</li>"
    )


def _render_assumptions_card(
    payload: Mapping[str, object],
    report_chip: ChipSpec | None,
) -> list[str]:
    """Render Assumptions, Hardware Specs, Constants Catalog, and L1 Glossary."""
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
    reserve_gb = max(0.0, _as_float(payload.get("hbm_reserve_gb"), 0.0))
    reserve_items = (
        [
            "      <li><strong>HBM Workspace Reserve:</strong> "
            f"<code>{reserve_gb:.2f} GB/core</code> subtracted for runtime/XLA "
            "workspace.</li>",
            "      <li><strong>Omitted in Layer-1 Scope:</strong> "
            "Empirical kernel utilization derates and trace-level LRU cache "
            "eviction or arrival burstiness.</li>",
        ]
        if reserve_gb > 0.0
        else [
            "      <li><strong>Omitted in Layer-1 Scope:</strong> "
            "Empirical kernel utilization derates, XLA temporary workspace, "
            "and trace-level LRU cache eviction or arrival burstiness.</li>",
        ]
    )
    provenance_item = _render_provenance_item(payload)
    provenance_items = [provenance_item] if provenance_item else []
    assumption_items = _render_solver_assumptions_items(payload)
    return [
        '  <section class="card assumptions-card">',
        "    <h2>Assumptions, Hardware Specs &amp; Scope</h2>",
        "    <ul>",
        "      <li><strong>Layer-1 Bounds vs. Estimates:</strong> "
        "<code>[bound]</code> metrics (<code>bound_tpot_ms</code>, "
        "<code>bound_ttft_ms</code>, <code>floor_throughput_per_chip</code>) are "
        "physical limits under full overlap; <code>[est]</code> metrics are "
        "uncalibrated serial-phase analytical estimates and do not prune "
        "physically legal layouts.</li>",
        chip_specs_item,
        *reserve_items,
        *provenance_items,
        *assumption_items,
        "    </ul>",
        "    <details><summary>Constants Catalog "
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
        '    <details class="glossary"><summary>'
        "Glossary of Physical Units, Submodule Axes &amp; Verdicts</summary>"
        "<ul>"
        + _render_glossary_unit_item(payload, report_chip)
        + "<li><strong>Attention vs. MoE Submodule Axes:</strong> "
        "<code>Attn: TP / DP / PCP / DCP</code> partitions attention heads, "
        "request replicas, prefill sequence chunks, and KV cache; "
        "<code>MoE: EP</code> shards whole routed experts across those same "
        "cores, while <code>MoE: TP</code> slices every expert by width across "
        "each TP group. <code>PP</code> splits whole transformer layers across "
        "stages.</li>"
        "<li><strong>3-Valued SLO Verdicts:</strong> "
        "<code>Est. Meets</code> (estimate &le; SLO), "
        "<code>Undecided</code> (physical lower bound &le; SLO &lt; estimate; "
        "kept for measurement), and <code>Ruled Out</code> "
        "(physical lower bound &gt; SLO). With no SLO set the column reads "
        "<code>No SLO</code>.</li>"
        "<li><strong>Binding:</strong> the resource that sets the hard step floor "
        "at the batch the estimate runs at: <code>hbm</code>, <code>matmul</code>, "
        "<code>vector</code>, or the collective bound (<code>chain</code> on a "
        "torus slice, <code>ici</code> otherwise). <code>dec:X,mix:Y</code> gives "
        "it for the decode-only step and for the mixed prefill and decode step. "
        "The bound columns and the step chart use the fewest resident sequences, "
        "so the chart's largest bar can name a different resource.</li>"
        + _render_glossary_pool_floor_item(payload)
        + "<li><strong>Tput Ceiling [bound]:</strong> "
        "<code>floor_throughput_per_chip</code>, the most tokens per second per "
        "chip the layout can physically produce at that concurrency.</li>"
        "<li><strong>KV Cap / Prefix Hit Rate:</strong> KV cache tokens that fit in "
        "the free HBM, and the prefix hit rate after clamping to that "
        "capacity.</li>"
        "<li><strong>Prefix-Hit Regime [est]:</strong> from an uncalibrated "
        "estimate sweep over prefix hit rates: the hit-rate range in which the "
        "first estimate-ordered layout stays first, and the crossover rate where "
        "it changes.</li>"
        "</ul></details>",
        "  </section>",
    ]


def _render_header_meta_suffix(
    payload: Mapping[str, object],
    *,
    chips_per_host: int,
) -> str:
    """Render budget, pins, and full dtype provenance (including state) for header."""
    chip_budget = _as_mapping(payload.get("chip_budget"))
    budget_source = str(
        chip_budget.get("budget_source") or payload.get("budget_source") or "flag"
    )
    budget_chips = _as_int(
        payload.get("budget"),
        _as_int(payload.get("target_chips"), _as_int(chip_budget.get("max_chips"))),
    )
    budget_hosts = count_hosts(budget_chips, chips_per_host=chips_per_host)
    budget_text = (
        f"Search Budget: {_format_chip_count(budget_chips)} / "
        f"{format_host_count(budget_hosts)} ({budget_source})"
        if budget_chips > 0
        else ""
    )
    pins = _as_mapping(chip_budget.get("pins", payload.get("pins"))) or {}
    pin_parts: list[str] = []
    if pins.get("prefill_topology"):
        pin_parts.append(f"P-slice: {pins['prefill_topology']}")
    if pins.get("decode_topology"):
        pin_parts.append(f"D-slice: {pins['decode_topology']}")
    pd_ratio_raw = pins.get("pd_ratio")
    if isinstance(pd_ratio_raw, str) and pd_ratio_raw:
        pin_parts.append(f"P:D={pd_ratio_raw}")
    elif (
        isinstance(pd_ratio_raw, Sequence)
        and not isinstance(pd_ratio_raw, (str, bytes))
        and len(pd_ratio_raw) >= PREFIX_RANGE_PAIR_LEN
    ):
        pin_parts.append(f"P:D={pd_ratio_raw[0]}:{pd_ratio_raw[1]}")
    elif isinstance(pins.get("pd"), str) and pins.get("pd"):
        pin_parts.append(f"P:D={pins['pd']}")
    pin_text = f" [Pins: {', '.join(pin_parts)}]" if pin_parts else ""

    dtypes_map = _as_mapping(payload.get("dtypes")) or {}
    dtypes_text = ""
    if dtypes_map:
        dense_dtype = dtypes_map.get("w_dense_dtype", "dense")
        dense_source = dtypes_map.get("w_dense_source", "default")
        dense_bytes = _format_dtype_bytes(dtypes_map.get("w_dense", 0.0))
        expert_dtype = dtypes_map.get("w_expert_dtype", "expert")
        expert_source = dtypes_map.get("w_expert_source", "default")
        expert_bytes = _format_dtype_bytes(dtypes_map.get("w_expert", 0.0))
        kv_dtype = dtypes_map.get("kv_dtype", "kv")
        kv_source = dtypes_map.get("kv_source", "default")
        kv_bytes = _format_dtype_bytes(dtypes_map.get("kv", 0.0))
        state_dtype = dtypes_map.get("state_dtype", "float32")
        state_source = dtypes_map.get("state_source", "default")
        state_bytes = _format_dtype_bytes(dtypes_map.get("state", 4.0))
        act_dtype = dtypes_map.get("act", "?")
        dtypes_text = (
            f"Dtypes: dense={dense_dtype} ({dense_bytes}, {dense_source}), "
            f"expert={expert_dtype} ({expert_bytes}, {expert_source}), "
            f"kv={kv_dtype} ({kv_bytes}, {kv_source}), "
            f"state={state_dtype} ({state_bytes}, {state_source}), "
            f"act={act_dtype}"
        )

    suffix = (
        f" &bull; <strong>{_escape(budget_text)}{_escape(pin_text)}</strong>"
        if budget_text
        else ""
    )
    if dtypes_text:
        suffix += f" &bull; <span>{_escape(dtypes_text)}</span>"
    return suffix


def _render_warnings_html(warnings: Sequence[str]) -> str:
    """Render deduplicated warning alert banners."""
    if not warnings:
        return ""
    seen: set[str] = set()
    parts: list[str] = []
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        parts.append(
            f'<div class="alert-box alert-warning"><strong>Warning:</strong> '
            f"{_escape(warning)}</div>"
        )
    return "\n  ".join(parts)


def _render_single_host_fit_card(
    payload: Mapping[str, object],
    *,
    has_moe: bool,
) -> str:
    """Render single-host slice HBM-legality verdict card (mixed-mode searches)."""
    single_host_fit = _as_mapping(payload.get("single_host_fit"))
    if not single_host_fit or str(payload.get("mode") or "") == "disagg":
        return ""
    chip_spec = _as_mapping(payload.get("chip_spec")) or {}
    topology = str(single_host_fit.get("topology") or "2x2x1")
    chips_per_host = _as_int(
        single_host_fit.get("chips_per_host"),
        _as_int(chip_spec.get("chips_per_host"), CHIPS["tpu7x"].chips_per_host),
    )
    cores_per_chip = _as_int(
        chip_spec.get("cores_per_chip"), CHIPS["tpu7x"].cores_per_chip
    )
    cores_per_host = _as_int(
        single_host_fit.get("cores_per_host"), chips_per_host * cores_per_chip
    )
    is_legal = bool(single_host_fit.get("feasible", False))
    raw_status = str(single_host_fit.get("status", "REJECTED"))
    display_status = "HBM LEGAL (NOT RULED OUT)" if is_legal else raw_status
    fit_reason = str(single_host_fit.get("reason", ""))
    hbm_used_gb = _as_float(single_host_fit.get("hbm_used_gb"))
    hbm_limit_gb = _as_float(single_host_fit.get("hbm_limit_gb"))
    submod_best = single_host_fit.get("best_submodule_parallelism") if has_moe else None
    raw_best = (
        submod_best
        if submod_best
        else single_host_fit.get("best_parallelism", f"TP{cores_per_host}")
    )
    best_parallelism = "N/A" if raw_best is None else str(raw_best)
    bg_color = "#dcfce7" if is_legal else "#fee2e2"
    text_color = "#166534" if is_legal else "#991b1b"
    return (
        '  <section id="single-host-fit" class="card">\n'
        f"    <h2>Single-Host {_escape(topology)} Physical Fit "
        f"(1 host, {_format_chip_count(chips_per_host)}, "
        f"{cores_per_host} TensorCores)</h2>\n"
        '    <div style="font-size:14px; margin-top:6px;">\n'
        f'      <span class="badge" style="background:{bg_color}; '
        f'color:{text_color}; font-weight:bold;">{_escape(display_status)}</span> '
        "&bull; \n"
        f"      <strong>{_escape(fit_reason)}</strong> &bull; \n"
        f"      HBM Footprint: <code>{_fixed(hbm_used_gb, _HBM_GB_DIGITS)} GB</code> / "
        f"<code>{_fixed(hbm_limit_gb, _HBM_GB_DIGITS)} GB</code> limit &bull; \n"
        f"      Sample Single-Host Layout: "
        f"<code>{_escape(best_parallelism)}</code>\n"
        "    </div>\n"
        "  </section>"
    )


def _funnel_last_stage_count(
    search: Mapping[str, object],
    *,
    has_slo: bool,
) -> int | None:
    """Return the funnel's last-stage count, or None without a search block."""
    if not search:
        return None
    if "not_ruled_out" in search:
        return _as_int(search.get("not_ruled_out"))
    return _as_int(search.get("slo_met" if has_slo else "hbm_fit"))


def _reference_rank_text(ref_report: Mapping[str, object]) -> str:
    """Describe a found reference layout's TPOT bound rank among distinct layouts."""
    rank = ref_report.get("rank_by_tpot_bound")
    if not isinstance(rank, int):
        return "Survived (L1 legal)"
    peers = ref_report.get("candidates")
    concurrency = ref_report.get("concurrency")
    of_peers = f" of {peers:,} layouts" if isinstance(peers, int) else ""
    at_concurrency = f" at c={concurrency}" if isinstance(concurrency, int) else ""
    return f"TPOT bound rank {rank:,}{of_peers}{at_concurrency} (ties share a rank)"


def _render_funnel_section(payload: Mapping[str, object]) -> list[str]:
    """Render strictly 4-stage physical Layer-1 Search Funnel HTML section."""
    search = _as_mapping(payload.get("search"))
    scenario = _as_mapping(payload.get("scenario"))
    enumerated_count = _as_int(search.get("enumerated"))
    core_grid_count = max(
        enumerated_count, _as_int(search.get("core_grid"), enumerated_count)
    )
    hbm_fit_count = _as_int(search.get("hbm_fit"))
    slo_met_count = _as_int(search.get("slo_met"))
    has_slo = (
        scenario.get("ttft_slo_ms") is not None
        or scenario.get("tpot_slo_ms") is not None
    )
    slo_step_class = (
        "funnel-step funnel-step-primary"
        if has_slo
        else "funnel-step funnel-step-muted"
    )
    if "not_ruled_out" in search:
        slo_stage_count = _funnel_last_stage_count(search, has_slo=has_slo) or 0
        ruled_out_count = max(0, hbm_fit_count - slo_stage_count)
        undecided_count = max(0, slo_stage_count - slo_met_count)
        slo_label = "Not Ruled Out (L1 Bound)" if has_slo else "SLO Bound Check"
        slo_breakdown_html = (
            '<div class="funnel-sub">'
            f"Ruled Out (bound &gt; SLO): {ruled_out_count:,}<br>"
            f"Est. Meets: {slo_met_count:,} &middot; Undecided: {undecided_count:,}"
            "</div>"
            if has_slo
            else '<div class="funnel-sub">No SLO set (all HBM-fit layouts kept)</div>'
        )
    else:
        slo_stage_count = _funnel_last_stage_count(search, has_slo=has_slo) or 0
        slo_label = "Not Ruled Out (L1 Bound)" if has_slo else "SLO Bound Check"
        slo_breakdown_html = (
            f'<div class="funnel-sub">{max(0, hbm_fit_count - slo_met_count):,} '
            "ruled out by latency SLO</div>"
            if has_slo
            else '<div class="funnel-sub">No SLO set (all HBM-fit layouts kept)</div>'
        )
    hbm_dropped = max(0, enumerated_count - hbm_fit_count)
    hbm_sub_html = (
        f'<div class="funnel-sub">{hbm_dropped:,} fail physical HBM fit '
        "(weights + logical KV + recurrent state)</div>"
        if hbm_dropped > 0
        else '<div class="funnel-sub">Fits physical HBM '
        "(weights + logical KV + recurrent state)</div>"
    )
    raw_regime = payload.get("activation_regime")
    funnel_heading = (
        f"Physical Pruning Funnel (MoE Activation: {_escape(raw_regime)})"
        if isinstance(raw_regime, str) and raw_regime
        else "Physical Pruning Funnel (Layer 1)"
    )
    envelope_feasible_html = (
        '        <div class="funnel-sub" style="margin-top:2px;">'
        f"{_as_int(search['envelope_feasible']):,} within activation envelope</div>"
        if has_slo and "envelope_feasible" in search
        else ""
    )
    return [
        '  <section id="search-funnel" class="card">',
        f"    <h2>{funnel_heading}</h2>",
        '    <div class="funnel-container">',
        '      <div class="funnel-step">',
        '        <div class="stat-label">1. Core-Valid Grid</div>',
        f'        <div class="stat-value">{core_grid_count:,}</div>',
        '        <div class="funnel-sub">Core-partitioned meshes &times; '
        "concurrencies</div>",
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        '      <div class="funnel-step">',
        '        <div class="stat-label">2. Physically Legal</div>',
        f'        <div class="stat-value">{enumerated_count:,}</div>',
        '        <div class="funnel-sub">Head, expert, layer &amp; '
        "core-product divisibility</div>",
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        '      <div class="funnel-step">',
        '        <div class="stat-label">3. HBM Fit</div>',
        f'        <div class="stat-value">{hbm_fit_count:,}</div>',
        f"        {hbm_sub_html}",
        "      </div>",
        '      <div class="funnel-arrow">&rarr;</div>',
        f'      <div class="{slo_step_class}">',
        f'        <div class="stat-label">{slo_label}</div>',
        f'        <div class="stat-value">{slo_stage_count:,}</div>',
        f"        {slo_breakdown_html}",
        envelope_feasible_html,
        "      </div>",
        "    </div>",
        "  </section>",
    ]


def _render_crossover_stat_html(payload: Mapping[str, object]) -> str:
    """Render prefix-hit regime stat box when crossover sweep data exists."""
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
        '        <div class="stat-label">Prefix-Hit Regime [est]</div>'
        '        <div class="stat-value" style="font-size:15px;">'
        f"{_escape(range_txt)} "
        '<span style="font-size:12px; font-weight:normal; color:#64748b;">'
        f"({_escape(cross_txt)})</span></div>"
        "      </div>"
    )


def _render_physical_envelope_card(
    payload: Mapping[str, object],
    all_points: Sequence[Mapping[str, object]],
    *,
    recommended: Mapping[str, object] | None,
    rec_infeasible: bool,
    accept_len: float,
    default_limit_gb: float,
    not_ruled_out_total: int | None,
) -> str:
    """Render Card 2: Layer-1 Physical Envelope & Survival Range Strip."""
    surviving = [pt for pt in all_points if _point_verdict(pt) != SLO_VERDICT_RULED_OUT]
    rec_verdict = (
        _point_verdict(recommended)
        if recommended is not None
        else SLO_VERDICT_RULED_OUT
    )
    if rec_infeasible:
        status_badge = (
            '<span class="badge" style="background:#fee2e2;color:#991b1b;">'
            "INFEASIBLE</span>"
        )
    elif rec_verdict == SLO_VERDICT_UNDECIDED:
        status_badge = (
            '<span class="badge" style="background:#fef3c7;color:#92400e;">'
            "UNDECIDED (BOUND &le; SLO &lt; EST)</span>"
        )
    else:
        status_badge = (
            '<span class="badge" style="background:#dcfce7;color:#166534;">'
            "L1 PHYSICAL ENVELOPE</span>"
        )

    if surviving:
        tpot_bounds = [
            bt
            for pt in surviving
            if (bt := _point_tpot_bound_ms(pt, accept_len=accept_len)) > 0.0
        ]
        min_tpot_str = _fixed(min(tpot_bounds), _TPOT_MS_DIGITS) if tpot_bounds else "-"
        ceil_tputs = [
            ft for pt in surviving if (ft := _point_floor_tput(pt)) is not None
        ]
        max_ceil_str = f"{max(ceil_tputs):,.1f}" if ceil_tputs else "-"
        min_hbm = min(_point_max_hbm_gb(pt) for pt in surviving)
        min_hbm_str = f"{min_hbm:.1f} / {default_limit_gb:.1f}"
        conc_set = sorted({_as_int(pt.get("concurrency")) for pt in surviving})
        conc_text = (
            "1 concurrency" if len(conc_set) == 1 else f"{len(conc_set)} concurrencies"
        )
        if not_ruled_out_total is None:
            surv_meta = (
                f"{len(surviving):,} layout-concurrency points carried in the "
                f"payload are not ruled out, across {conc_text}"
            )
        else:
            total = max(not_ruled_out_total, len(surviving))
            surv_meta = (
                f"{total:,} layout-concurrency points not ruled out across {conc_text}"
            )
            if len(surviving) < total:
                surv_meta += (
                    "; the stats below cover the layouts carried in this report's "
                    "payload"
                )
    else:
        min_tpot_str = "-"
        max_ceil_str = "-"
        min_hbm_str = "N/A"
        surv_meta = "N/A"

    ref_report = _as_mapping(payload.get("reference"))
    if ref_report:
        if bool(ref_report.get("found")):
            ref_status = _reference_rank_text(ref_report)
        else:
            ref_status = f"Not enumerated ({ref_report.get('rejected_by', 'pruned')})"
    else:
        ref_status = "Not specified"

    crossover_html = _render_crossover_stat_html(payload)
    return "\n".join(
        [
            '  <section id="physical-envelope" class="card">',
            "    <h2>Physical Envelope &amp; Bounds Summary (Layer 1)</h2>",
            '    <div style="font-size:13.5px; color:#334155;">',
            f"      {status_badge} &bull; {_escape(surv_meta)}",
            "    </div>",
            '    <div class="grid-stats">',
            '      <div class="stat-box">',
            '        <div class="stat-label">Min Bound TPOT [bound]</div>',
            f'        <div class="stat-value">{min_tpot_str} '
            '<span style="font-size:12px;font-weight:normal;color:#64748b;">'
            "ms</span></div>",
            "      </div>",
            '      <div class="stat-box">',
            '        <div class="stat-label">Max Ceiling Tput [bound]</div>',
            f'        <div class="stat-value">{max_ceil_str} '
            '<span style="font-size:12px;font-weight:normal;color:#64748b;">'
            "tok/s/chip</span></div>",
            "      </div>",
            '      <div class="stat-box">',
            '        <div class="stat-label">Min HBM Footprint [bound]</div>',
            f'        <div class="stat-value">{min_hbm_str} '
            '<span style="font-size:12px;font-weight:normal;color:#64748b;">'
            "GB</span></div>",
            "      </div>",
            '      <div class="stat-box">',
            '        <div class="stat-label">Reference Layout</div>',
            '        <div class="stat-value" style="font-size:14px;'
            f'overflow-wrap:anywhere;">{_escape(ref_status)}</div>',
            "      </div>",
            crossover_html,
            "    </div>",
            "  </section>",
        ]
    )


def _format_bound_family_label(
    entry: Mapping[str, object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> str:
    """Format concise mesh family label from a bound_families_by_concurrency entry."""
    raw_label = entry.get("family_label")
    if isinstance(raw_label, str) and raw_label:
        return raw_label
    raw_family = entry.get("family")
    if isinstance(raw_family, Sequence) and not isinstance(raw_family, (str, bytes)):
        parts = [str(part) for part in raw_family if part]
        if parts:
            return " | ".join(parts)
    mode = str(entry.get("mode") or "")
    pref = _as_mapping(entry.get("prefill_family"))
    dec = _as_mapping(entry.get("decode_family"))
    if pref and dec:
        return (
            f"P: {_pool_inline_label(pref, has_moe=has_moe, chip=chip)} / "
            f"D: {_pool_inline_label(dec, has_moe=has_moe, chip=chip)}"
        )
    if dec:
        return _pool_inline_label(dec, has_moe=has_moe, chip=chip)
    return mode or "-"


def _format_bound_family_members(
    members: Sequence[object],
    *,
    has_moe: bool = False,
    chip: ChipSpec | None = None,
) -> list[str]:
    """Extract compact human-readable sample labels from family members."""
    labels: list[str] = []
    for member in members[:_MAX_FAMILY_SAMPLE_LABELS]:
        if isinstance(member, Mapping) and "decode_pool" in member:
            labels.append(_pools_display(member, has_moe=has_moe, chip=chip))
        elif isinstance(member, Mapping) and isinstance(member.get("pool"), Mapping):
            labels.append(
                _pool_inline_label(
                    _as_mapping(member.get("pool")), has_moe=has_moe, chip=chip
                )
            )
        elif isinstance(member, Mapping) and "tp" in member:
            labels.append(_pool_inline_label(member, has_moe=has_moe, chip=chip))
        elif isinstance(member, str) and member:
            labels.append(member)
    return labels


def _compact_bound_families(
    raw_families: Mapping[str, object],
    *,
    has_moe: bool,
    chip: ChipSpec | None,
) -> dict[str, list[dict[str, object]]]:
    """Strip nested full ParetoPoint member blobs from bound families."""
    compact: dict[str, list[dict[str, object]]] = {}
    for conc_key, raw_entries in raw_families.items():
        rows: list[dict[str, object]] = []
        for idx, raw_entry in enumerate(_as_sequence(raw_entries), 1):
            entry = _as_mapping(raw_entry)
            if not entry:
                continue
            members = _as_sequence(entry.get("members"))
            sample_labels = _format_bound_family_members(
                members, has_moe=has_moe, chip=chip
            )
            member_count = _as_int(entry.get("member_count"), len(members))
            rows.append(
                {
                    "rank": _as_int(entry.get("rank"), idx),
                    "family_label": _format_bound_family_label(
                        entry, has_moe=has_moe, chip=chip
                    ),
                    "mode": str(entry.get("mode") or ""),
                    "member_count": member_count,
                    "min_bound_tpot_ms": _as_opt_float(entry.get("min_bound_tpot_ms")),
                    "min_bound_ttft_ms": _as_opt_float(entry.get("min_bound_ttft_ms")),
                    "min_envelope_tpot_ms": _as_opt_float(
                        entry.get("min_envelope_tpot_ms")
                    ),
                    "min_envelope_ttft_ms": _as_opt_float(
                        entry.get("min_envelope_ttft_ms")
                    ),
                    "binding": str(entry.get("binding") or ""),
                    "slo_verdict": str(entry.get("slo_verdict") or ""),
                    "sample_members": sample_labels,
                }
            )
        compact[str(conc_key)] = rows[:_MAX_FAMILY_ROWS_PER_CONC]
    return compact


def _compact_pool_records(
    records: Sequence[object],
    *,
    has_moe: bool,
    chip: ChipSpec | None,
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
) -> list[dict[str, object]]:
    """Compact standalone prefill/decode pool records for tabbed explorer display."""
    out: list[dict[str, object]] = []
    for raw in records:
        rec = _as_mapping(raw)
        pool = _as_mapping(rec.get("pool"))
        if not pool:
            continue
        main, sub = _pool_cell_parts(pool, has_moe=has_moe, chip=chip)
        bt_val = _as_opt_float(rec.get("tpot_floor_ms"))
        btt_val = _as_opt_float(rec.get("ttft_floor_ms"))
        if bool(rec.get("slo_met", True)):
            verdict = SLO_VERDICT_ESTIMATE_MEETS
        else:
            tpot_ruled = (
                tpot_slo_ms is not None and bt_val is not None and bt_val > tpot_slo_ms
            )
            ttft_ruled = (
                ttft_slo_ms is not None
                and btt_val is not None
                and btt_val > ttft_slo_ms
            )
            verdict = (
                SLO_VERDICT_RULED_OUT
                if (tpot_ruled or ttft_ruled)
                else SLO_VERDICT_UNDECIDED
            )
        out.append(
            {
                "c": _as_int(rec.get("concurrency")),
                "r": str(rec.get("role") or pool.get("role") or ""),
                "l": main,
                "s": sub,
                "bt": bt_val,
                "btt": btt_val,
                "et": _as_float(rec.get("token_throughput_per_chip")),
                "ei": _as_opt_float(rec.get("interactivity_tok_s")),
                "etp": _as_opt_float(rec.get("tpot_ms")),
                "ett": _as_opt_float(rec.get("ttft_ms")),
                "hbm": round(_as_float(pool.get("hbm_used_gb")), _HBM_GB_DIGITS),
                "lim": round(_as_float(pool.get("hbm_limit_gb")), _HBM_GB_DIGITS),
                "b": str(rec.get("binding") or ""),
                "v": verdict,
            }
        )
    return out


def _select_pool_rows(
    records: Sequence[dict[str, object]],
    *,
    floor_key: str,
) -> list[dict[str, object]]:
    """Keep each concurrency's first pools by floor, lowest floor first.

    Ruled-out pools sort last; ties go to the higher throughput estimate.
    """
    by_conc: dict[int, list[dict[str, object]]] = {}
    for rec in records:
        by_conc.setdefault(_as_int(rec.get("c")), []).append(rec)

    def _order(rec: Mapping[str, object]) -> tuple[float, float, float]:
        floor_ms = _as_opt_float(rec.get(floor_key))
        return (
            1.0 if rec.get("v") == SLO_VERDICT_RULED_OUT else 0.0,
            floor_ms if floor_ms is not None and floor_ms > 0.0 else math.inf,
            -_as_float(rec.get("et")),
        )

    selected: list[dict[str, object]] = []
    for conc in sorted(by_conc):
        selected.extend(sorted(by_conc[conc], key=_order)[:_POOL_ROWS_PER_CONC])
    return selected


def _explorer_sort_keys(
    point: Mapping[str, object],
    *,
    accept_len: float,
) -> tuple[tuple[float, ...], ...]:
    """Return one key per explorer sort option, in the inline script's order."""
    ruled = 1.0 if _point_verdict(point) == SLO_VERDICT_RULED_OUT else 0.0
    bound_tpot = round(
        _point_tpot_bound_ms(point, accept_len=accept_len), _TPOT_MS_DIGITS
    )
    bound_tpot = bound_tpot if bound_tpot > 0.0 else math.inf
    ceiling = round(_point_floor_tput(point) or 0.0, _RATE_DIGITS)
    est_tput = round(_as_float(point.get("token_throughput_per_chip")), _RATE_DIGITS)
    est_inter = round(_interactivity_val(point), _RATE_DIGITS)
    hbm_gb = round(_point_max_hbm_gb(point), 2)
    return (
        (ruled, bound_tpot, -est_tput),
        (ruled, -ceiling, bound_tpot),
        (ruled, -est_tput, -est_inter),
        (ruled, -est_inter, -est_tput),
        (ruled, hbm_gb, bound_tpot),
    )


def _select_explorer_points(
    group: Sequence[Mapping[str, object]],
    *,
    ref_key: tuple[object, ...] | None,
    accept_len: float,
) -> list[Mapping[str, object]]:
    """Pick one concurrency's embedded rows, ordered by TPOT bound.

    Always keeps the reference layout and the first rows of every sort option, so
    each sort shows its exact leaders. Pareto-marked points, then bound order,
    fill the rest up to the row cap.
    """
    keyed = [(_explorer_sort_keys(pt, accept_len=accept_len), pt) for pt in group]
    chosen: dict[tuple[object, ...], Mapping[str, object]] = {}

    def _take(point: Mapping[str, object], *, capped: bool) -> None:
        if not capped or len(chosen) < _MAX_EXPLORER_ROWS_PER_CONC:
            chosen.setdefault(_point_dedup_key(point), point)

    if ref_key is not None:
        for _, pt in keyed:
            if _point_dedup_key(pt) == ref_key:
                _take(pt, capped=False)
    num_sort_keys = len(keyed[0][0]) if keyed else 0
    for key_idx in range(num_sort_keys):
        leaders = sorted(keyed, key=lambda item, idx=key_idx: item[0][idx])
        for _, pt in leaders[:_EXPLORER_ROWS_PER_SORT_KEY]:
            _take(pt, capped=False)
    by_bound = sorted(keyed, key=lambda item: item[0][0])
    for _, pt in by_bound:
        if bool(pt.get("is_pareto")):
            _take(pt, capped=True)
    for _, pt in by_bound:
        _take(pt, capped=True)
    return [pt for _, pt in by_bound if _point_dedup_key(pt) in chosen]


def _compact_explorer_rows(
    all_points: Sequence[Mapping[str, object]],
    *,
    ref_key: tuple[object, ...] | None,
    accept_len: float,
    has_moe: bool,
    chip: ChipSpec | None,
    default_limit_gb: float,
) -> list[dict[str, object]]:
    """Serialize compact per-concurrency candidate rows for client-side explorer."""
    by_conc: dict[int, list[Mapping[str, object]]] = {}
    for pt in all_points:
        by_conc.setdefault(_as_int(pt.get("concurrency")), []).append(pt)

    compact_rows: list[dict[str, object]] = []
    for conc in sorted(by_conc):
        selected = _select_explorer_points(
            by_conc[conc], ref_key=ref_key, accept_len=accept_len
        )
        for pt in selected:
            key = _point_dedup_key(pt)
            pref = _as_mapping(pt.get("prefill_pool"))
            dec = _as_mapping(pt.get("decode_pool"))
            p_main, p_sub = (
                _pool_cell_parts(pref, has_moe=has_moe, chip=chip) if pref else ("", "")
            )
            d_main, d_sub = _pool_cell_parts(dec, has_moe=has_moe, chip=chip)
            df = _as_mapping(pt.get("decode_floor"))
            df_vec = (
                [
                    round(_as_float(df.get("hbm_bound_ms", df.get("hbm_ms"))), 3),
                    round(_as_float(df.get("matmul_bound_ms", df.get("matmul_ms"))), 3),
                    round(_as_float(df.get("vector_bound_ms")), 3),
                    round(_as_float(df.get("coll_bound_ms", df.get("ici_ms"))), 3),
                    round(_as_float(df.get("hard_floor_ms")), 3),
                    round(_as_float(df.get("chain_ms")), 3),
                    round(_as_float(df.get("hop_latency_ms")), 3),
                ]
                if df
                else None
            )
            mem_pools: list[list[object]] = []
            for pool in (pref, dec):
                if not pool:
                    continue
                mem_pools.append(
                    [
                        str(pool.get("role") or "pool"),
                        str(pool.get("topology") or ""),
                        round(
                            max(0.0, _as_float(pool.get("hbm_weights_gb"))),
                            _HBM_GB_DIGITS,
                        ),
                        round(
                            max(0.0, _as_float(pool.get("hbm_kv_gb"))), _HBM_GB_DIGITS
                        ),
                        round(
                            max(0.0, _as_float(pool.get("hbm_state_gb"))),
                            _HBM_GB_DIGITS,
                        ),
                        round(
                            max(0.0, _as_float(pool.get("hbm_used_gb"))), _HBM_GB_DIGITS
                        ),
                        round(
                            max(
                                0.0,
                                _as_float(pool.get("hbm_limit_gb"), default_limit_gb),
                            ),
                            _HBM_GB_DIGITS,
                        ),
                    ]
                )
            kv_cap = _as_int(
                pt.get("kv_cache_capacity_tokens", pt.get("kv_capacity_tokens"))
            )
            eff_hit = _as_opt_float(pt.get("effective_prefix_hit_rate"))
            kv_cap_txt = (
                f"{kv_cap // 1000:,}k"
                if kv_cap >= 1000
                else (str(kv_cap) if kv_cap > 0 else "-")
            )
            hit_txt = f"{eff_hit * PERCENT_SCALE:.0f}%" if eff_hit is not None else "-"
            ft_val = _point_floor_tput(pt)
            compact_rows.append(
                {
                    "c": conc,
                    "m": str(pt.get("mode") or ""),
                    "ch": _as_int(pt.get("total_chips")),
                    "pm": p_main,
                    "ps": p_sub,
                    "dm": d_main,
                    "ds": d_sub,
                    "v": _point_verdict(pt),
                    "bt": round(
                        _point_tpot_bound_ms(pt, accept_len=accept_len),
                        _TPOT_MS_DIGITS,
                    ),
                    "btt": round(_point_ttft_bound_ms(pt), _TTFT_MS_DIGITS),
                    "fc": round(ft_val, _RATE_DIGITS) if ft_val is not None else None,
                    "ei": round(_interactivity_val(pt), _RATE_DIGITS),
                    "et": round(
                        _as_float(pt.get("token_throughput_per_chip")), _RATE_DIGITS
                    ),
                    "etp": round(_tpot_val(pt), _EST_TPOT_MS_DIGITS),
                    "ett": round(_ttft_val(pt), _TTFT_MS_DIGITS),
                    "kv": f"{kv_cap_txt} / {hit_txt}",
                    "b": str(pt.get("binding") or ""),
                    "hbm": round(_point_max_hbm_gb(pt), 2),
                    "ref": ref_key is not None and key == ref_key,
                    "df": df_vec,
                    "mp": mem_pools,
                }
            )
    return compact_rows


_MEM_POOL_USED_IDX = 5
_PAIRED_POOL_COUNT = 2


def _pool_sub_with_hbm(sub: object, mem_pool: object) -> str:
    """Append a pool's HBM use (GB per core) to its slice summary line."""
    text = str(sub or "")
    fields = _as_sequence(mem_pool)
    if len(fields) <= _MEM_POOL_USED_IDX:
        return text
    hbm_txt = (
        f"{_fixed(_as_float(fields[_MEM_POOL_USED_IDX]), _HBM_GB_DIGITS)} GB/core HBM"
    )
    return f"{text} · {hbm_txt}" if text else hbm_txt


def _render_static_explorer_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    is_disagg: bool,
    selected_idx: int,
    has_slo: bool,
) -> str:
    """Render initial static HTML <tr> rows for the default concurrency."""
    if not rows:
        colspan = 13 if is_disagg else 12
        return (
            f'<tr><td colspan="{colspan}" style="text-align:center;'
            'color:var(--text-muted);">'
            "No legal candidates at this concurrency</td></tr>"
        )
    indices = list(range(min(_DEFAULT_VISIBLE_ROWS, len(rows))))
    for idx, row in enumerate(rows):
        if bool(row.get("ref")) and idx not in indices:
            indices.append(idx)
    parts: list[str] = []
    for idx in indices:
        row = rows[idx]
        row_cls = ' class="row-selected"' if idx == selected_idx else ""
        ref_tag = ' <span class="v-badge v-ref">REF</span>' if row.get("ref") else ""
        mem_pools = _as_sequence(row.get("mp"))
        prefill_mem = mem_pools[0] if len(mem_pools) >= _PAIRED_POOL_COUNT else None
        decode_mem = mem_pools[-1] if mem_pools else None
        pm_txt = _escape(row.get("pm") or "-")
        ps_txt = _escape(_pool_sub_with_hbm(row.get("ps"), prefill_mem))
        dm_txt = _escape(row.get("dm") or "-")
        ds_txt = _escape(_pool_sub_with_hbm(row.get("ds"), decode_mem))
        if is_disagg:
            pool_cells = (
                f'<td class="pool-cell-disagg"><div class="pool-main">'
                f"{pm_txt}{ref_tag}</div>"
                f'<div class="pool-sub">{ps_txt}</div></td>'
                f'<td class="pool-cell-disagg"><div class="pool-main">{dm_txt}</div>'
                f'<div class="pool-sub">{ds_txt}</div></td>'
            )
        else:
            pool_cells = (
                f'<td class="pool-cell"><div class="pool-main">{dm_txt}{ref_tag}</div>'
                f'<div class="pool-sub">{ds_txt}</div></td>'
            )
        verdict = str(row.get("v") or SLO_VERDICT_ESTIMATE_MEETS)
        bound_tpot = _format_opt_ms(row.get("bt"), _TPOT_MS_DIGITS)
        bound_ttft = _format_opt_ms(row.get("btt"), _TTFT_MS_DIGITS)
        est_inter = _fixed(_as_float(row.get("ei")), _RATE_DIGITS)
        est_tpot = _format_opt_ms(row.get("etp"), _EST_TPOT_MS_DIGITS)
        parts.append(
            f'<tr data-row-idx="{idx}"{row_cls}>'
            f'<td class="num-cell">{idx + 1}</td>'
            f'<td class="num-cell">{_as_int(row.get("ch"))}</td>'
            f"{pool_cells}"
            f"<td>{_verdict_badge_html(verdict, has_slo=has_slo)}</td>"
            f'<td class="num-cell"><strong>{bound_tpot}'
            "</strong></td>"
            f'<td class="num-cell">{bound_ttft}</td>'
            f'<td class="num-cell"><strong>{_format_opt_rate(row.get("fc"))}'
            "</strong></td>"
            f'<td class="num-cell">{est_inter}</td>'
            f'<td class="num-cell">{_format_opt_rate(row.get("et"))}</td>'
            f'<td class="num-cell">{est_tpot}</td>'
            f'<td class="num-cell">{_escape(row.get("kv") or "- / -")}</td>'
            f'<td class="binding-cell"><code>{_escape(row.get("b") or "-")}</code></td>'
            "</tr>"
        )
    return "\n        ".join(parts)


def _render_static_pool_rows(
    records: Sequence[Mapping[str, object]],
    *,
    default_conc: int | None,
    has_slo: bool,
) -> str:
    """Render one pool table's static rows for the default concurrency."""
    shown = [
        rec
        for rec in records
        if default_conc is None or _as_int(rec.get("c")) == default_conc
    ]
    if not shown:
        return (
            '<tr><td colspan="9" style="text-align:center;'
            'color:var(--text-muted);">No pool records at this concurrency</td></tr>'
        )
    parts: list[str] = []
    for idx, rec in enumerate(shown[:_DEFAULT_VISIBLE_ROWS], 1):
        verdict = str(rec.get("v") or SLO_VERDICT_ESTIMATE_MEETS)
        tpot_floor = _format_opt_ms(rec.get("bt"), _TPOT_MS_DIGITS)
        ttft_floor = _format_opt_ms(rec.get("btt"), _TTFT_MS_DIGITS)
        parts.append(
            "<tr>"
            f'<td class="num-cell">{idx}</td>'
            f'<td class="num-cell"><code>c={_as_int(rec.get("c"))}</code></td>'
            f'<td class="pool-cell"><div class="pool-main">'
            f"{_escape(rec.get('l') or '-')}</div>"
            f'<div class="pool-sub">{_escape(rec.get("s") or "")}</div></td>'
            f"<td>{_verdict_badge_html(verdict, has_slo=has_slo)}</td>"
            f'<td class="num-cell">{tpot_floor}</td>'
            f'<td class="num-cell">{ttft_floor}</td>'
            f'<td class="num-cell">{_format_opt_rate(rec.get("et"))}</td>'
            f'<td class="num-cell">{_fixed(_as_float(rec.get("hbm")), 1)} / '
            f"{_fixed(_as_float(rec.get('lim')), 1)} GB</td>"
            f'<td class="binding-cell"><code>{_escape(rec.get("b") or "-")}</code></td>'
            "</tr>"
        )
    return "\n        ".join(parts)


def _render_static_family_rows(
    families_by_conc: Mapping[str, list[dict[str, object]]],
    default_conc: int | None,
) -> str:
    """Render the mesh-family rows the inline script shows in the default state."""
    if default_conc is None:
        # The script walks object keys: integer keys ascending, then the rest.
        keys = sorted(
            families_by_conc, key=lambda key: (0, int(key)) if key.isdigit() else (1, 0)
        )
    else:
        keys = [str(default_conc)] if str(default_conc) in families_by_conc else []
    entries = [(key, entry) for key in keys for entry in families_by_conc[key]]
    if not entries:
        return (
            '<tr><td colspan="8" style="text-align:center;'
            'color:var(--text-muted);">No mesh families at this concurrency</td></tr>'
        )
    parts: list[str] = []
    for conc_key, entry in entries[:_DEFAULT_VISIBLE_ROWS]:
        tpot_s = _format_opt_ms(entry.get("min_bound_tpot_ms"), _TPOT_MS_DIGITS)
        ttft_s = _format_opt_ms(entry.get("min_bound_ttft_ms"), _TTFT_MS_DIGITS)
        env_t = _format_opt_ms(entry.get("min_envelope_tpot_ms"), _TPOT_MS_DIGITS)
        env_tt = _format_opt_ms(entry.get("min_envelope_ttft_ms"), _TTFT_MS_DIGITS)
        samples = [str(s) for s in _as_sequence(entry.get("sample_members"))]
        cnt = _as_int(entry.get("member_count"))
        extra = max(0, cnt - len(samples))
        if samples:
            shown_samples = [*samples, f"+{extra} more"] if extra > 0 else samples
            members_txt = f"{cnt} ({', '.join(shown_samples)})"
        else:
            members_txt = str(cnt)
        parts.append(
            "<tr>"
            f"<td><code>c={_escape(conc_key)}</code></td>"
            f"<td>#{_as_int(entry.get('rank'))}</td>"
            f"<td><code>{_escape(entry.get('family_label') or '-')}</code></td>"
            f"<td><strong>{tpot_s}</strong></td>"
            f"<td>{ttft_s}</td>"
            f"<td>{env_t} / {env_tt}</td>"
            f"<td><code>{_escape(entry.get('binding') or '-')}</code></td>"
            f"<td>{_escape(members_txt)}</td>"
            "</tr>"
        )
    return "\n        ".join(parts)


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
  padding: 20px 24px;
}
.container { max-width: 1380px; margin: 0 auto; }
header {
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}
h1 { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
h2 { font-size: 17px; font-weight: 600; margin-bottom: 10px; color: #1e293b; }
.meta { font-size: 13px; color: var(--text-muted); line-height: 1.6; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11.5px;
  font-weight: 600;
  background-color: var(--primary-light);
  color: var(--primary);
  margin-right: 6px;
}
.badge-fact { background-color: #dcfce7; color: #166534; }
.badge-derived { background-color: #e0f2fe; color: #0369a1; }
.badge-assumed { background-color: #fef3c7; color: #92400e; }
.badge-config { background-color: #f3e8ff; color: #6b21a8; }
.v-badge {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}
.v-meets { background: #dcfce7; color: #166534; }
.v-undecided { background: #fef3c7; color: #92400e; }
.v-ruled { background: #f1f5f9; color: #64748b; }
.v-none { background: #f1f5f9; color: #475569; }
.legend-swatch {
  display: inline-block;
  width: 11px;
  height: 11px;
  border-radius: 2px;
  margin: 0 5px 0 14px;
  vertical-align: -1px;
}
.v-ref { background: #f3e8ff; color: #6b21a8; margin-left: 4px; }
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  overflow-x: auto;
}
.grid-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.stat-box {
  background: var(--bg-secondary);
  padding: 10px 12px;
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
  font-size: 18px;
  font-weight: 700;
  color: var(--text-main);
  margin-top: 2px;
}
.alert-box {
  padding: 10px 14px;
  border-radius: 6px;
  margin-bottom: 14px;
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
  align-items: stretch;
  margin-top: 8px;
}
.funnel-step {
  flex: 1;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  text-align: center;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.funnel-step-primary {
  border-color: var(--primary);
  background: var(--primary-light);
}
.funnel-step-muted {
  background: #f8fafc;
  border-style: dashed;
  opacity: 0.85;
}
.funnel-arrow {
  color: var(--text-muted);
  font-size: 16px;
  display: flex;
  align-items: center;
}
.funnel-sub {
  font-size: 11.5px;
  color: var(--text-muted);
  margin-top: 3px;
  line-height: 1.35;
}
.controls-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
}
.pill-group { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.pill-btn {
  padding: 5px 11px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: #ffffff;
  color: #334155;
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}
.pill-btn:hover { background: #f1f5f9; }
.pill-btn.active {
  background: var(--primary);
  border-color: var(--primary);
  color: #ffffff;
}
.filter-input, .sort-select {
  padding: 5px 10px;
  border-radius: 6px;
  border: 1px solid #cbd5e1;
  font-size: 12.5px;
  background: #ffffff;
  color: #0f172a;
}
.filter-input { min-width: 220px; }
.chart-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.chart-container {
  display: flex;
  justify-content: center;
  margin: 8px 0;
}
.chart-svg {
  width: 100%;
  max-width: 940px;
  height: auto;
  font-family: inherit;
}
.chart-legend {
  color: var(--text-muted);
  font-size: 12px;
  margin-top: 6px;
  text-align: center;
}
.section-note {
  color: var(--text-muted);
  font-size: 12.5px;
  margin-top: 4px;
}
.table-wrap { overflow-x: auto; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-top: 8px;
  table-layout: auto;
}
th, td {
  padding: 6px 6px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
th {
  background: var(--bg-secondary);
  font-weight: 600;
  color: #475569;
  font-size: 11px;
  line-height: 1.25;
  white-space: normal;
}
tbody tr { cursor: pointer; }
tbody tr:hover td { background: #f1f5f9; }
tbody tr.row-selected td { background: #eff6ff; }
.pool-cell { min-width: 215px; white-space: normal; }
.pool-cell-disagg { min-width: 170px; white-space: normal; }
.num-cell { white-space: nowrap; font-variant-numeric: tabular-nums; }
.binding-cell { white-space: nowrap; }
.pool-main { font-weight: 600; color: #0f172a; font-size: 12px; line-height: 1.3; }
.pool-sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.hidden-panel { display: none; }
details.glossary {
  margin-top: 10px;
  border-top: 1px solid var(--border);
  padding-top: 8px;
  font-size: 12px;
  color: var(--text-muted);
}
details.glossary > summary {
  cursor: pointer;
  font-weight: 600;
  color: #475569;
  font-size: 12.5px;
}
details.glossary ul { margin: 6px 0 6px 18px; line-height: 1.5; }
footer {
  margin-top: 20px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
  font-size: 11.5px;
  color: var(--text-muted);
  text-align: center;
}
"""

_JS_TEMPLATE = """
(function() {
  var dataEl = document.getElementById("roofline-data");
  if (!dataEl) return;
  var payload;
  try {
    payload = JSON.parse(dataEl.textContent || "{}");
  } catch (e) {
    return;
  }
  var ex = payload.explorer || {};
  var rows = ex.rows || [];
  var families = payload.bound_families_by_concurrency || {};
  var prefillPools = ex.prefill_pools || [];
  var decodePools = ex.decode_pools || [];
  var isDisagg = Boolean(ex.is_disagg);
  var hasSlo = Boolean(ex.has_slo);
  var NICE_STEPS = __AXIS_NICE_STEPS__;
  var AXIS_HEADROOM = __AXIS_HEADROOM__;
  var VISIBLE_ROWS = __VISIBLE_ROWS__;
  var MAX_TICK_DECIMALS = __MAX_TICK_DECIMALS__;
  var FIXED_MAX_VALUE = __FIXED_MAX_VALUE__;
  var TPOT_MS_DIGITS = __TPOT_MS_DIGITS__;
  var TTFT_MS_DIGITS = __TTFT_MS_DIGITS__;
  var EST_TPOT_MS_DIGITS = __EST_TPOT_MS_DIGITS__;
  var RATE_DIGITS = __RATE_DIGITS__;
  var HBM_GB_DIGITS = __HBM_GB_DIGITS__;
  var ENVELOPE_TIE_TOLERANCE = __ENVELOPE_TIE_TOLERANCE__;
  var TICK_STEP_REL_TOL = __TICK_STEP_REL_TOL__;
  var legend = ex.legend || {};
  var defaultLimit = Number(ex.default_hbm_limit_gb || 103.0);
  var state = {
    conc: ex.default_conc != null ? String(ex.default_conc) : "all",
    plotMode: "est",
    sortBy: "bt",
    filter: "",
    showAll: false,
    tab: "candidates",
    selectedIdx: typeof ex.default_selected_idx === "number"
      ? ex.default_selected_idx : 0
  };
  var SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, text) {
    var n = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function(k) {
        n.setAttribute(k, String(attrs[k]));
      });
    }
    if (text != null) n.textContent = String(text);
    return n;
  }
  function svgEl(tag, attrs, text) {
    var n = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function(k) {
        n.setAttribute(k, String(attrs[k]));
      });
    }
    if (text != null) n.textContent = String(text);
    return n;
  }
  function fmtFixed(v, d, grouped) {
    // Half-up on the shortest decimal form; values clamp to [0, FIXED_MAX_VALUE].
    v = v > 0 ? Math.min(v, FIXED_MAX_VALUE) : 0;
    var sci = String(v).split("e");
    var scaled = String(Math.round(Number(sci[0] + "e" + (Number(sci[1] || 0) + d))));
    // A scaled value of 1e21 or more prints with an exponent; no cell or tick
    // reaches it (ticks that large have 0 decimals).
    var out = (scaled.indexOf("e") === -1 ? Number(scaled + "e-" + d) : v).toFixed(d);
    if (!grouped) return out;
    var parts = out.split("."), head = parts[0], tail = "";
    while (head.length > 3) {
      tail = "," + head.slice(-3) + tail;
      head = head.slice(0, -3);
    }
    parts[0] = head + tail;
    return parts.join(".");
  }
  function fmtMs(v, d) {
    return (typeof v === "number" && v > 0) ? fmtFixed(v, d) : "-";
  }
  function fmtRate(v) {
    if (typeof v !== "number" || v <= 0) return "-";
    return fmtFixed(v, 1, true);
  }
  function niceMax(raw) {
    if (!isFinite(raw) || raw <= 0) return 1;
    var step = (raw * AXIS_HEADROOM) / 4;
    var sc = Math.pow(10, Math.floor(Math.log10(step)));
    var fr = step / sc;
    var ns = NICE_STEPS[NICE_STEPS.length - 1];
    for (var i = 0; i < NICE_STEPS.length; i++) {
      if (fr <= NICE_STEPS[i]) { ns = NICE_STEPS[i]; break; }
    }
    return ns * sc * 4;
  }
  function tickDecimals(step) {
    for (var d = 0; d < MAX_TICK_DECIMALS; d++) {
      if (Math.abs(Number(step.toFixed(d)) - step) <=
          TICK_STEP_REL_TOL * Math.abs(step)) return d;
    }
    return MAX_TICK_DECIMALS;
  }
  function fmtTick(v, step) {
    if (v <= 0) return "0";
    return fmtFixed(v, tickDecimals(step), true);
  }
  function verdictBadge(v) {
    if (!hasSlo) return el("span", {"class": "v-badge v-none"}, "No SLO");
    var cls = v === "estimate_meets"
      ? "v-meets" : (v === "undecided" ? "v-undecided" : "v-ruled");
    var lbl = v === "estimate_meets"
      ? "Est. Meets" : (v === "undecided" ? "Undecided" : "Ruled Out");
    return el("span", {"class": "v-badge " + cls}, lbl);
  }
  function rowLabel(r) {
    if (!r) return "";
    return r.pm ? ("P: " + r.pm + " / D: " + r.dm) : (r.dm || "");
  }
  function subWithHbm(sub, memPool) {
    if (!memPool || memPool.length < 6) return sub || "";
    var txt = fmtFixed(memPool[5], HBM_GB_DIGITS) + " GB/core HBM";
    return sub ? (sub + " · " + txt) : txt;
  }
  function activeRows() {
    var q = state.filter.trim().toLowerCase();
    var out = [];
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (state.conc !== "all" && String(r.c) !== state.conc) continue;
      if (q) {
        var hay = (rowLabel(r) + " " + r.b + " " + r.v + " c=" + r.c).toLowerCase();
        if (hay.indexOf(q) === -1) continue;
      }
      out.push({idx: i, r: r});
    }
    out.sort(function(a, b) {
      var ra = a.r, rb = b.r;
      var va = ra.v === "ruled_out" ? 1 : 0;
      var vb = rb.v === "ruled_out" ? 1 : 0;
      if (va !== vb) return va - vb;
      if (state.sortBy === "bt") {
        return ((ra.bt || 1e9) - (rb.bt || 1e9)) || (rb.et - ra.et);
      }
      if (state.sortBy === "fc") {
        return ((rb.fc || 0) - (ra.fc || 0)) || ((ra.bt || 1e9) - (rb.bt || 1e9));
      }
      if (state.sortBy === "et") return (rb.et - ra.et) || (rb.ei - ra.ei);
      if (state.sortBy === "ei") return (rb.ei - ra.ei) || (rb.et - ra.et);
      return (ra.hbm - rb.hbm) || ((ra.bt || 1e9) - (rb.bt || 1e9));
    });
    return out;
  }
  function renderLimits(row) {
    var wrap = document.getElementById("limits-svg-wrap");
    var meta = document.getElementById("selected-layout-limits-meta");
    if (!wrap) return;
    if (meta) {
      meta.textContent = row
        ? ("Selected: " + rowLabel(row) + " (c=" + row.c + ")")
        : "";
    }
    wrap.textContent = "";
    var W = 920, H = 345, PL = 175, PR = 115, PT = 28, PW = W - PL - PR;
    var s = svgEl("svg", {viewBox: "0 0 " + W + " " + H, "class": "chart-svg"});
    if (!row || !row.df) {
      s.appendChild(svgEl(
        "text",
        {x: W / 2, y: H / 2, "text-anchor": "middle", fill: "#64748b", "font-size": 14},
        "No decode floor data available"
      ));
      wrap.appendChild(s);
      return;
    }
    var df = row.df;
    var bItems = [
      ["HBM Bound", df[0], "#3b82f6"],
      ["MatMul Bound", df[1], "#10b981"],
      ["Vector Bound", df[2], "#06b6d4"],
      ["Collective Bound", df[3], "#f59e0b"],
      ["Hard Floor", df[4], "#ef4444"]
    ];
    var eItems = [
      ["Serial Op Overhead", df[5], "#8b5cf6"],
      ["Hop Latency", df[6], "#ec4899"]
    ];
    var mx = 1e-6;
    bItems.concat(eItems).forEach(function(it) { if (it[1] > mx) mx = it[1]; });
    mx *= 1.16;
    s.appendChild(svgEl(
      "text",
      {x: PL, y: 16, "font-size": 11, "font-weight": 700, fill: "#0f172a"},
      "PHYSICAL LOWER BOUNDS [bound_*]"
    ));
    function drawRow(y, it) {
      var v = it[1], bw = v < 5e-4 ? 0 : Math.max(2.5, (v / mx) * PW);
      var wt = it[0] === "Hard Floor" ? 700 : 500;
      s.appendChild(svgEl(
        "text",
        {x: PL - 14, y: y + 15, "text-anchor": "end", "font-size": 12.5,
         "font-weight": wt, fill: "#1e293b"},
        it[0]
      ));
      var rc = svgEl("rect", {
        x: PL, y: y, width: bw === 0 ? "0" : bw.toFixed(1),
        height: 22, fill: it[2], rx: 4
      });
      rc.appendChild(svgEl("title", null, it[0] + ": " + v.toFixed(3) + " ms"));
      s.appendChild(rc);
      s.appendChild(svgEl(
        "text",
        {x: (PL + bw + 10).toFixed(1), y: y + 15, "font-size": 12,
         "font-weight": wt, fill: "#334155"},
        v.toFixed(3) + " ms"
      ));
    }
    var y = PT;
    bItems.forEach(function(it) { drawRow(y, it); y += 34; });
    var dy = y + 6;
    s.appendChild(svgEl("line", {
      x1: PL, y1: dy, x2: W - 24, y2: dy,
      stroke: "#cbd5e1", "stroke-dasharray": "4,4"
    }));
    s.appendChild(svgEl(
      "text",
      {x: PL, y: dy + 18, "font-size": 11, "font-weight": 700, fill: "#6b21a8"},
      "SERIAL ESTIMATE OVERHEAD [est_*]"
    ));
    y = dy + 28;
    eItems.forEach(function(it) { drawRow(y, it); y += 34; });
    wrap.appendChild(s);
  }
  function renderMemory(row) {
    var wrap = document.getElementById("memory-svg-wrap");
    var meta = document.getElementById("selected-layout-memory-meta");
    if (!wrap) return;
    if (meta) {
      meta.textContent = row
        ? ("Selected: " + rowLabel(row) + " (c=" + row.c + ")")
        : "";
    }
    wrap.textContent = "";
    var pools = (row && row.mp) ? row.mp : [];
    var W = 920, PL = 175, PR = 185, PT = 22, BH = 26, RH = 62, PW = W - PL - PR;
    var H = PT + Math.max(1, pools.length) * RH + 8;
    var s = svgEl("svg", {viewBox: "0 0 " + W + " " + H, "class": "chart-svg"});
    if (!pools.length) {
      s.appendChild(svgEl(
        "text",
        {x: W / 2, y: 60, "text-anchor": "middle", fill: "#64748b", "font-size": 14},
        "No pool memory data available"
      ));
      wrap.appendChild(s);
      return;
    }
    pools.forEach(function(p, i) {
      var y = PT + i * RH, lim = p[6] > 0 ? p[6] : defaultLimit;
      var used = p[5], free = Math.max(0, lim - used);
      var lbl = (p[0].charAt(0).toUpperCase() + p[0].slice(1)) + " (" + p[1] + ")";
      s.appendChild(svgEl(
        "text",
        {x: PL - 14, y: y + 18, "text-anchor": "end", "font-size": 13,
         "font-weight": 500, fill: "#1e293b"},
        lbl
      ));
      var segs = [
        [p[2], "#3b82f6", "Weights"],
        [p[3], "#10b981", "KV Cache"],
        [p[4], "#f59e0b", "Recurrent State"],
        [free, "#e2e8f0", "Headroom"]
      ];
      var cx = PL;
      segs.forEach(function(sg) {
        if (lim <= 0 || sg[0] <= 0) return;
        var rem = Math.max(0, PL + PW - cx);
        var w = sg[1] === "#e2e8f0" ? rem : Math.min(rem, (sg[0] / lim) * PW);
        if (w <= 0) return;
        var rc = svgEl("rect", {
          x: cx.toFixed(1), y: y, width: w.toFixed(1), height: BH, fill: sg[1]
        });
        var segTitle = sg[2] + ": " + fmtFixed(sg[0], HBM_GB_DIGITS) + " GB";
        rc.appendChild(svgEl("title", null, segTitle));
        s.appendChild(rc);
        cx += w;
      });
      s.appendChild(svgEl("rect", {
        x: PL, y: y, width: PW, height: BH, fill: "none", stroke: "#94a3b8", rx: 4
      }));
      var pct = lim > 0 ? Math.round((used / lim) * 100) : 0;
      s.appendChild(svgEl(
        "text",
        {x: PL + PW + 12, y: y + 18, "font-size": 12.5,
         "font-weight": 600, fill: "#334155"},
        fmtFixed(used, HBM_GB_DIGITS) + " / " + fmtFixed(lim, HBM_GB_DIGITS) +
        " GB (" + pct + "%)"
      ));
    });
    wrap.appendChild(s);
  }
  function placeLabels(anchors, minX, maxX, minY, maxY) {
    var placed = [], out = [];
    var offs = [
      [12, -24], [-12, -24], [12, 10], [-12, 10], [0, -34],
      [0, 22], [24, -38], [-24, -38], [24, 28], [-24, 28]
    ];
    anchors.slice(0, 4).forEach(function(a) {
      var bw = Math.min(210, a.lbl.length * 6.6 + 10), bh = 18;
      for (var k = 0; k < offs.length; k++) {
        var dx = offs[k][0], dy = offs[k][1];
        var bx = dx >= 0 ? a.px + dx : a.px + dx - bw;
        var by = a.py + dy;
        bx = Math.max(minX + 4, Math.min(maxX - bw - 4, bx));
        by = Math.max(minY + 4, Math.min(maxY - bh - 4, by));
        var hit = false;
        for (var j = 0; j < placed.length; j++) {
          var o = placed[j];
          if (bx < o[2] + 4 && bx + bw + 4 > o[0] &&
              by < o[3] + 4 && by + bh + 4 > o[1]) {
            hit = true;
            break;
          }
        }
        if (!hit) {
          placed.push([bx, by, bx + bw, by + bh]);
          out.push({px: a.px, py: a.py, bx: bx, by: by, bw: bw,
                    lbl: a.lbl, color: a.color});
          break;
        }
      }
    });
    return out;
  }
  function renderPlot(items) {
    var wrap = document.getElementById("tradeoff-svg-wrap");
    var leg = document.getElementById("tradeoff-legend");
    var viewNote = document.getElementById("tradeoff-view-note");
    if (viewNote) {
      var scopeLbl = state.conc === "all"
        ? "All concurrencies" : ("c=" + state.conc);
      var cnt = items.length;
      var noun = cnt === 1 ? "layout" : "layouts";
      if (!cnt) {
        viewNote.textContent = "Current view (" + scopeLbl + "): 0 layouts.";
      } else {
        var seenCoords = {};
        var distinct = 0;
        items.forEach(function(it) {
          var r = it.r;
          var xk = state.plotMode === "bound" ? (r.bt || 0) : (r.ei || 0);
          var yk = state.plotMode === "bound"
            ? (r.fc != null ? r.fc : (r.et || 0)) : (r.et || 0);
          var key = String(Number(xk)) + "|" + String(Number(yk));
          if (!seenCoords[key]) { seenCoords[key] = true; distinct += 1; }
        });
        if (distinct < cnt) {
          var posNoun = distinct === 1 ? "position" : "positions";
          viewNote.textContent = "Current view (" + scopeLbl + "): " +
            fmtFixed(cnt, 0, true) + " embedded " + noun + " (" +
            fmtFixed(distinct, 0, true) + " distinct plot " + posNoun +
            "; layouts with identical coordinates overlap).";
        } else {
          viewNote.textContent = "Current view (" + scopeLbl + "): " +
            fmtFixed(cnt, 0, true) + " embedded " + noun + ".";
        }
      }
    }
    if (!wrap) return;
    wrap.textContent = "";
    var W = 920, H = 420, PL = 86, PR = 36, PT = 26, PB = 54;
    var PW = W - PL - PR, PH = H - PT - PB;
    var s = svgEl("svg", {viewBox: "0 0 " + W + " " + H, "class": "chart-svg"});
    if (!items.length) {
      s.appendChild(svgEl(
        "text",
        {x: W / 2, y: H / 2, "text-anchor": "middle", fill: "#64748b", "font-size": 14},
        rows.length ? "No matching candidates" : "__NO_LAYOUTS_TEXT__"
      ));
      if (leg) leg.textContent = legend[state.plotMode] || "";
      wrap.appendChild(s);
      return;
    }
    var pts = items.map(function(it) {
      var r = it.r;
      var x = state.plotMode === "bound" ? (r.bt > 0 ? 1000 / r.bt : 0) : r.ei;
      var y = state.plotMode === "bound" ? (r.fc != null ? r.fc : r.et) : r.et;
      return {x: Math.max(0, x), y: Math.max(0, y), r: r, idx: it.idx};
    });
    var mx = niceMax(Math.max.apply(null, pts.map(function(p) { return p.x; })));
    var my = niceMax(Math.max.apply(null, pts.map(function(p) { return p.y; })));
    function sx(v) { return PL + (v / mx) * PW; }
    function sy(v) { return PT + PH - (v / my) * PH; }
    s.appendChild(svgEl("rect", {
      x: PL, y: PT, width: PW, height: PH,
      fill: "#f8fafc", stroke: "#e2e8f0", rx: 4
    }));
    for (var t = 0; t <= 4; t++) {
      var fr = t / 4, gx = PL + fr * PW, gy = PT + PH - fr * PH;
      s.appendChild(svgEl("line", {
        x1: gx.toFixed(1), y1: PT, x2: gx.toFixed(1), y2: PT + PH,
        stroke: "#e2e8f0", "stroke-dasharray": "3,3"
      }));
      s.appendChild(svgEl("line", {
        x1: PL, y1: gy.toFixed(1), x2: PL + PW, y2: gy.toFixed(1),
        stroke: "#e2e8f0", "stroke-dasharray": "3,3"
      }));
      s.appendChild(svgEl("text", {
        x: gx.toFixed(1), y: H - 28, "text-anchor": "middle",
        "font-size": 11.5, fill: "#64748b"
      }, fmtTick(fr * mx, mx / 4)));
      s.appendChild(svgEl("text", {
        x: PL - 10, y: (gy + 4).toFixed(1), "text-anchor": "end",
        "font-size": 11.5, fill: "#64748b"
      }, fmtTick(fr * my, my / 4)));
    }
    var xLbl = state.plotMode === "bound"
      ? "Max Bound Interactivity (1000 / bound_tpot_ms, tok/s)"
      : "Est. Interactivity (tok/s)";
    var yLbl = state.plotMode === "bound"
      ? "Physical Throughput Ceiling (tok/s/chip)"
      : "Est. Token Throughput per Chip (tok/s/chip)";
    s.appendChild(svgEl("text", {
      x: (PL + PW / 2).toFixed(1), y: H - 8, "text-anchor": "middle",
      "font-size": 12.5, "font-weight": 600, fill: "#334155"
    }, xLbl));
    var ym = (PT + PH / 2).toFixed(1);
    s.appendChild(svgEl("text", {
      x: 20, y: ym, "text-anchor": "middle", "font-size": 12.5,
      "font-weight": 600, fill: "#334155",
      transform: "rotate(-90 20 " + ym + ")"
    }, yLbl));
    var surv = pts.filter(function(p) { return p.r.v !== "ruled_out"; });
    var env = surv.slice().sort(function(a, b) { return b.x - a.x || b.y - a.y; });
    var hull = [], bestY = -1;
    env.forEach(function(p) {
      if (p.y > bestY + ENVELOPE_TIE_TOLERANCE) { hull.push(p); bestY = p.y; }
    });
    hull.sort(function(a, b) { return a.x - b.x; });
    if (hull.length >= 2) {
      var pStr = hull.map(function(p) {
        return sx(p.x).toFixed(1) + "," + sy(p.y).toFixed(1);
      }).join(" ");
      s.appendChild(svgEl("polyline", {
        points: pStr, fill: "none",
        stroke: state.plotMode === "bound" ? "#0284c7" : "#2563eb",
        "stroke-width": 2, "stroke-dasharray": "6,4", opacity: 0.8
      }));
    }
    pts.forEach(function(p) {
      var r = p.r, cx = sx(p.x), cy = sy(p.y);
      var fill = r.v === "ruled_out"
        ? "#94a3b8" : (r.v === "undecided" ? "#fef3c7" : "#2563eb");
      var st = r.v === "ruled_out"
        ? "#64748b" : (r.v === "undecided" ? "#d97706" : "#1d4ed8");
      var sel = p.idx === state.selectedIdx;
      var c = svgEl("circle", {
        cx: cx.toFixed(1), cy: cy.toFixed(1), r: sel ? 6.5 : 4.5,
        fill: fill, stroke: sel ? "#0f172a" : st,
        "stroke-width": sel ? 2 : 1.2,
        opacity: r.v === "ruled_out" ? 0.42 : 0.85, style: "cursor:pointer"
      });
      c.appendChild(svgEl(
        "title", null,
        rowLabel(r) + " | c=" + r.c + " | " + (hasSlo ? r.v : "no SLO") +
        " | TPOT bound=" + fmtMs(r.bt, TPOT_MS_DIGITS) + " ms | Ceiling=" +
        fmtRate(r.fc) + " tok/s/chip | Est=" + fmtRate(r.et) + " tok/s/chip, " +
        fmtFixed(r.ei, RATE_DIGITS) + " tok/s"
      ));
      c.addEventListener("click", function() {
        state.selectedIdx = p.idx;
        renderAll();
      });
      s.appendChild(c);
    });
    var pool = surv.length ? surv : pts;
    var anchors = [], seenA = {};
    function addAnchor(p, lbl, color, dash) {
      if (!p || seenA[p.idx]) return;
      seenA[p.idx] = true;
      var px = sx(p.x), py = sy(p.y);
      var ringAttrs = {
        cx: px.toFixed(1), cy: py.toFixed(1), r: 9.5,
        fill: "none", stroke: color, "stroke-width": 2.2
      };
      if (dash) ringAttrs["stroke-dasharray"] = "3,2";
      s.appendChild(svgEl("circle", ringAttrs));
      anchors.push({px: px, py: py, lbl: lbl, color: color});
    }
    var refP = pts.find(function(p) { return p.r.ref; });
    addAnchor(refP, "Reference", "#7c3aed", true);
    if (state.plotMode === "bound") {
      var minBtP = pool.reduce(function(a, b) {
        return ((b.r.bt || 1e9) < (a.r.bt || 1e9)) ? b : a;
      }, pool[0]);
      addAnchor(minBtP, "Min Bound TPOT", "#059669", false);
      if (pool.some(function(p) { return p.r.fc != null; })) {
        var maxFcP = pool.reduce(function(a, b) {
          return ((b.r.fc || 0) > (a.r.fc || 0)) ? b : a;
        }, pool[0]);
        addAnchor(maxFcP, "Max Ceiling Tput", "#0284c7", false);
      }
    } else {
      var maxEtP = pool.reduce(function(a, b) {
        return (b.r.et > a.r.et) ? b : a;
      }, pool[0]);
      addAnchor(maxEtP, "Max Est. Tput", "#d97706", false);
    }
    placeLabels(anchors, PL, PL + PW, PT, PT + PH).forEach(function(lb) {
      s.appendChild(svgEl("line", {
        x1: lb.px.toFixed(1), y1: lb.py.toFixed(1),
        x2: (lb.bx + lb.bw / 2).toFixed(1), y2: (lb.by + 9).toFixed(1),
        stroke: lb.color, "stroke-width": 1, opacity: 0.65
      }));
      s.appendChild(svgEl("rect", {
        x: lb.bx.toFixed(1), y: lb.by.toFixed(1),
        width: lb.bw.toFixed(1), height: 18, rx: 4,
        fill: "#ffffff", "fill-opacity": 0.94,
        stroke: lb.color, "stroke-width": 1.2
      }));
      s.appendChild(svgEl("text", {
        x: (lb.bx + 5).toFixed(1), y: (lb.by + 12.5).toFixed(1),
        "font-size": 11, "font-weight": 600, fill: lb.color
      }, lb.lbl));
    });
    if (leg) leg.textContent = legend[state.plotMode] || "";
    wrap.appendChild(s);
  }
  function renderTable(items) {
    var tbody = document.getElementById("explorer-tbody");
    if (!tbody) return items.length;
    tbody.textContent = "";
    var visible = [];
    items.forEach(function(it, pos) {
      if (state.showAll || pos < VISIBLE_ROWS || it.r.ref) {
        visible.push({idx: it.idx, r: it.r, pos: pos});
      }
    });
    if (!visible.length) {
      var tr0 = el("tr");
      tr0.appendChild(el(
        "td",
        {colspan: isDisagg ? 13 : 12, style: "text-align:center;color:#64748b"},
        "No matching layouts"
      ));
      tbody.appendChild(tr0);
      return items.length;
    }
    visible.forEach(function(it) {
      var r = it.r;
      var memPools = r.mp || [];
      var cls = it.idx === state.selectedIdx ? {"class": "row-selected"} : null;
      var tr = el("tr", cls);
      tr.addEventListener("click", function() {
        state.selectedIdx = it.idx;
        renderAll();
      });
      tr.appendChild(el("td", {"class": "num-cell"}, it.pos + 1));
      tr.appendChild(el("td", {"class": "num-cell"}, r.ch));
      if (isDisagg) {
        var tdP = el("td", {"class": "pool-cell-disagg"});
        var pm = el("div", {"class": "pool-main"}, r.pm || "-");
        if (r.ref) pm.appendChild(el("span", {"class": "v-badge v-ref"}, "REF"));
        tdP.appendChild(pm);
        tdP.appendChild(el(
          "div", {"class": "pool-sub"},
          subWithHbm(r.ps, memPools.length >= 2 ? memPools[0] : null)
        ));
        tr.appendChild(tdP);
      }
      var tdD = el("td", {"class": isDisagg ? "pool-cell-disagg" : "pool-cell"});
      var dm = el("div", {"class": "pool-main"}, r.dm || "-");
      if (!isDisagg && r.ref) {
        dm.appendChild(el("span", {"class": "v-badge v-ref"}, "REF"));
      }
      tdD.appendChild(dm);
      tdD.appendChild(el(
        "div", {"class": "pool-sub"},
        subWithHbm(r.ds, memPools.length ? memPools[memPools.length - 1] : null)
      ));
      tr.appendChild(tdD);
      var tdV = el("td");
      tdV.appendChild(verdictBadge(r.v));
      tr.appendChild(tdV);
      var tdBt = el("td", {"class": "num-cell"});
      tdBt.appendChild(el("strong", null, fmtMs(r.bt, TPOT_MS_DIGITS)));
      tr.appendChild(tdBt);
      tr.appendChild(el("td", {"class": "num-cell"}, fmtMs(r.btt, TTFT_MS_DIGITS)));
      var tdFc = el("td", {"class": "num-cell"});
      tdFc.appendChild(el("strong", null, fmtRate(r.fc)));
      tr.appendChild(tdFc);
      tr.appendChild(el("td", {"class": "num-cell"}, fmtFixed(r.ei, RATE_DIGITS)));
      tr.appendChild(el("td", {"class": "num-cell"}, fmtRate(r.et)));
      tr.appendChild(el("td", {"class": "num-cell"}, fmtMs(r.etp, EST_TPOT_MS_DIGITS)));
      tr.appendChild(el("td", {"class": "num-cell"}, r.kv));
      var tdB = el("td", {"class": "binding-cell"});
      tdB.appendChild(el("code", null, r.b || "-"));
      tr.appendChild(tdB);
      tbody.appendChild(tr);
    });
    return items.length;
  }
  function renderFamilies() {
    var tbody = document.getElementById("families-tbody");
    if (!tbody) return 0;
    tbody.textContent = "";
    var keys = state.conc === "all"
      ? Object.keys(families) : (families[state.conc] ? [state.conc] : []);
    var allEntries = [];
    keys.forEach(function(k) {
      (families[k] || []).forEach(function(f) { allEntries.push({c: k, f: f}); });
    });
    var shown = state.showAll ? allEntries : allEntries.slice(0, VISIBLE_ROWS);
    if (!shown.length) {
      var tr0 = el("tr");
      tr0.appendChild(el(
        "td",
        {colspan: 8, style: "text-align:center;color:#64748b"},
        "No mesh families at this concurrency"
      ));
      tbody.appendChild(tr0);
      return 0;
    }
    shown.forEach(function(item) {
      var f = item.f, tr = el("tr");
      var tdC = el("td");
      tdC.appendChild(el("code", null, "c=" + item.c));
      tr.appendChild(tdC);
      tr.appendChild(el("td", null, "#" + f.rank));
      var tdF = el("td");
      tdF.appendChild(el("code", null, f.family_label || "-"));
      tr.appendChild(tdF);
      var tdBt = el("td");
      tdBt.appendChild(el("strong", null, fmtMs(f.min_bound_tpot_ms, TPOT_MS_DIGITS)));
      tr.appendChild(tdBt);
      tr.appendChild(el("td", null, fmtMs(f.min_bound_ttft_ms, TTFT_MS_DIGITS)));
      var envTxt = fmtMs(f.min_envelope_tpot_ms, TPOT_MS_DIGITS) + " / " +
        fmtMs(f.min_envelope_ttft_ms, TTFT_MS_DIGITS);
      tr.appendChild(el("td", null, envTxt));
      var tdB = el("td");
      tdB.appendChild(el("code", null, f.binding || "-"));
      tr.appendChild(tdB);
      var samples = f.sample_members || [];
      var cnt = f.member_count || 0;
      var extra = Math.max(0, cnt - samples.length);
      var memTxt = String(cnt);
      if (samples.length) {
        var shownSamples = extra > 0
          ? samples.concat(["+" + extra + " more"])
          : samples;
        memTxt += " (" + shownSamples.join(", ") + ")";
      }
      tr.appendChild(el("td", null, memTxt));
      tbody.appendChild(tr);
    });
    return allEntries.length;
  }
  function renderPoolBody(bodyId, list) {
    var tbody = document.getElementById(bodyId);
    if (!tbody) return 0;
    tbody.textContent = "";
    var filtered = list.filter(function(p) {
      return state.conc === "all" || String(p.c) === state.conc;
    });
    var shown = state.showAll ? filtered : filtered.slice(0, VISIBLE_ROWS);
    if (!shown.length) {
      var tr0 = el("tr");
      tr0.appendChild(el(
        "td",
        {colspan: 9, style: "text-align:center;color:#64748b"},
        "No pool records at this concurrency"
      ));
      tbody.appendChild(tr0);
      return 0;
    }
    shown.forEach(function(p, i) {
      var tr = el("tr");
      tr.appendChild(el("td", {"class": "num-cell"}, i + 1));
      var tdC = el("td", {"class": "num-cell"});
      tdC.appendChild(el("code", null, "c=" + p.c));
      tr.appendChild(tdC);
      var tdL = el("td", {"class": "pool-cell"});
      tdL.appendChild(el("div", {"class": "pool-main"}, p.l));
      tdL.appendChild(el("div", {"class": "pool-sub"}, p.s));
      tr.appendChild(tdL);
      var tdV = el("td");
      tdV.appendChild(verdictBadge(p.v));
      tr.appendChild(tdV);
      tr.appendChild(el("td", {"class": "num-cell"}, fmtMs(p.bt, TPOT_MS_DIGITS)));
      tr.appendChild(el("td", {"class": "num-cell"}, fmtMs(p.btt, TTFT_MS_DIGITS)));
      tr.appendChild(el("td", {"class": "num-cell"}, fmtRate(p.et)));
      var hbmTxt = fmtFixed(p.hbm, HBM_GB_DIGITS) + " / " +
        fmtFixed(p.lim, HBM_GB_DIGITS) + " GB";
      tr.appendChild(el("td", {"class": "num-cell"}, hbmTxt));
      var tdB = el("td", {"class": "binding-cell"});
      tdB.appendChild(el("code", null, p.b || "-"));
      tr.appendChild(tdB);
      tbody.appendChild(tr);
    });
    return filtered.length;
  }
  function syncTabs() {
    document.querySelectorAll("[data-conc-btn]").forEach(function(b) {
      b.classList.toggle("active", b.getAttribute("data-conc-btn") === state.conc);
    });
    document.querySelectorAll("[data-plot-btn]").forEach(function(b) {
      b.classList.toggle("active", b.getAttribute("data-plot-btn") === state.plotMode);
    });
    document.querySelectorAll("[data-tab-btn]").forEach(function(b) {
      b.classList.toggle("active", b.getAttribute("data-tab-btn") === state.tab);
    });
    var panels = {
      candidates: "panel-candidates",
      prefill: "panel-prefill",
      decode: "panel-decode",
      families: "bound-families-table"
    };
    Object.keys(panels).forEach(function(k) {
      var node = document.getElementById(panels[k]);
      if (node) node.classList.toggle("hidden-panel", state.tab !== k);
    });
    var rowControls = document.getElementById("explorer-row-controls");
    if (rowControls) {
      rowControls.style.display = state.tab === "candidates" ? "" : "none";
    }
  }
  function syncToggle(total) {
    var btn = document.getElementById("explorer-toggle-all");
    if (!btn) return;
    btn.textContent = state.showAll
      ? ("Show First " + VISIBLE_ROWS + " (" + total + " total)")
      : ("Show All (" + total + ")");
    btn.style.display = total > VISIBLE_ROWS ? "inline-block" : "none";
  }
  function renderAll() {
    syncTabs();
    var items = activeRows();
    if (items.length && !items.some(function(it) {
      return it.idx === state.selectedIdx;
    })) {
      state.selectedIdx = items[0].idx;
    }
    var selRow = (state.selectedIdx >= 0 && state.selectedIdx < rows.length)
      ? rows[state.selectedIdx] : null;
    renderPlot(items);
    var totals = {
      candidates: renderTable(items),
      families: renderFamilies(),
      prefill: renderPoolBody("prefill-pools-tbody", prefillPools),
      decode: renderPoolBody("decode-pools-tbody", decodePools)
    };
    syncToggle(totals[state.tab] || 0);
    renderLimits(selRow);
    renderMemory(selRow);
  }
  document.querySelectorAll("[data-conc-btn]").forEach(function(b) {
    b.addEventListener("click", function() {
      state.conc = b.getAttribute("data-conc-btn");
      state.showAll = false;
      renderAll();
    });
  });
  document.querySelectorAll("[data-plot-btn]").forEach(function(b) {
    b.addEventListener("click", function() {
      state.plotMode = b.getAttribute("data-plot-btn");
      renderAll();
    });
  });
  document.querySelectorAll("[data-tab-btn]").forEach(function(b) {
    b.addEventListener("click", function() {
      state.tab = b.getAttribute("data-tab-btn");
      renderAll();
    });
  });
  var sortEl = document.getElementById("explorer-sort");
  if (sortEl) sortEl.addEventListener("change", function() {
    state.sortBy = sortEl.value;
    renderAll();
  });
  var filterEl = document.getElementById("explorer-filter");
  if (filterEl) filterEl.addEventListener("input", function() {
    state.filter = filterEl.value;
    renderAll();
  });
  var toggleEl = document.getElementById("explorer-toggle-all");
  if (toggleEl) toggleEl.addEventListener("click", function() {
    state.showAll = !state.showAll;
    renderAll();
  });
  renderAll();
})();
"""

JS_INTERACTIVE = (
    _JS_TEMPLATE.replace("__AXIS_NICE_STEPS__", json.dumps(list(_AXIS_NICE_STEPS)))
    .replace("__AXIS_HEADROOM__", repr(_AXIS_HEADROOM_SCALE))
    .replace("__VISIBLE_ROWS__", str(_DEFAULT_VISIBLE_ROWS))
    .replace("__MAX_TICK_DECIMALS__", str(_MAX_TICK_DECIMALS))
    .replace("__FIXED_MAX_VALUE__", repr(_FIXED_MAX_VALUE))
    .replace("__TPOT_MS_DIGITS__", str(_TPOT_MS_DIGITS))
    .replace("__TTFT_MS_DIGITS__", str(_TTFT_MS_DIGITS))
    .replace("__EST_TPOT_MS_DIGITS__", str(_EST_TPOT_MS_DIGITS))
    .replace("__RATE_DIGITS__", str(_RATE_DIGITS))
    .replace("__HBM_GB_DIGITS__", str(_HBM_GB_DIGITS))
    .replace("__ENVELOPE_TIE_TOLERANCE__", repr(FRONTIER_2D_TIE_TOLERANCE))
    .replace("__NO_LAYOUTS_TEXT__", _NO_LAYOUTS_TEXT)
    .replace("__TICK_STEP_REL_TOL__", repr(_TICK_STEP_REL_TOL))
)


def _display_model_label(raw_model: object) -> str:
    """Return the model label for headers, dropping directories of a config path."""
    text = str(raw_model or "Model")
    if text.endswith(".json") or text.startswith(("/", "~", "./", "../")):
        return Path(text).name or text
    return text


def _tradeoff_legend_text(*, has_slo: bool, has_reference: bool) -> dict[str, str]:
    """Return the trade-off legend per plot mode; the inline script reads it too."""
    colors = (
        "Blue: Est. Meets · Amber: Undecided · Grey: Ruled Out"
        if has_slo
        else "No SLO set: every plotted layout passed the HBM fit"
    )
    est_rings = "Reference / Max Est. Tput" if has_reference else "Max Est. Tput"
    bound_rings = (
        "Reference / Min Bound TPOT / Max Ceiling Tput"
        if has_reference
        else "Min Bound TPOT / Max Ceiling Tput"
    )
    click_note = "Click any point or table row to inspect Step/HBM."
    return {
        "est": (
            "Dashed line: uncalibrated est_* upper envelope "
            f"(reference only; no layouts pruned) · {colors} · "
            f"Rings: {est_rings} · {click_note}"
        ),
        "bound": (
            "Dashed line: physical bound upper envelope "
            f"(reference only; no layouts pruned) · {colors} · "
            f"Rings: {bound_rings} · {click_note}"
        ),
    }


def _tradeoff_view_note(
    rows: Sequence[Mapping[str, object]],
    *,
    conc: int | str | None,
    mode: str,
) -> str:
    """Summarize plotted layout count and coordinate overlap for the active view."""
    scope = "All concurrencies" if conc is None or str(conc) == "all" else f"c={conc}"
    count = len(rows)
    noun = "layout" if count == 1 else "layouts"
    if count == 0:
        return f"Current view ({scope}): 0 layouts."
    coords: set[tuple[float, float]] = set()
    for row in rows:
        if mode == "bound":
            x_key = _as_float(row.get("bt"))
            y_key = (
                _as_float(row.get("fc"))
                if row.get("fc") is not None
                else _as_float(row.get("et"))
            )
        else:
            x_key = _as_float(row.get("ei"))
            y_key = _as_float(row.get("et"))
        coords.add((x_key, y_key))
    distinct = len(coords)
    if distinct < count:
        pos_noun = "position" if distinct == 1 else "positions"
        return (
            f"Current view ({scope}): {count:,} embedded {noun} "
            f"({distinct:,} distinct plot {pos_noun}; layouts with identical "
            "coordinates overlap)."
        )
    return f"Current view ({scope}): {count:,} embedded {noun}."


def _explorer_scope_note(
    shown: int,
    total: int,
    *,
    carried: int,
    total_from_funnel: bool,
    top_k_limit: int | None,
    ruled_out_shown: int,
    has_reference: bool,
    has_families: bool,
) -> str:
    """Describe which layout-concurrency points the report embeds and why.

    ``total`` is the funnel's last-stage count, or the payload's own count of
    points not ruled out (``carried``) when the payload has no funnel counters.
    """
    verb = "point is" if ruled_out_shown == 1 else "points are"
    if total == 0:
        if not ruled_out_shown:
            return "No layout-concurrency point is left to plot."
        return (
            "No layout-concurrency point passes the bound check. "
            f"The {ruled_out_shown:,} ruled-out {verb} plotted and listed."
        )
    ruled_note = (
        f" {ruled_out_shown:,} ruled-out {verb} listed after them."
        if ruled_out_shown
        else ""
    )
    scope = (
        "layout-concurrency points that are not ruled out"
        if total_from_funnel
        else "layout-concurrency points carried in the payload that are not ruled out"
    )
    if shown >= total:
        if total_from_funnel:
            return f"All {total:,} {scope} are plotted and listed.{ruled_note}"
        return f"Plots and lists the {total:,} {scope}.{ruled_note}"
    reasons: list[str] = []
    if carried < total:
        if top_k_limit is not None:
            kept = [
                f"the first {top_k_limit} layouts per concurrency by estimate order",
                f"the solver's top-{top_k_limit} list and recommended layout",
                "the Pareto-marked layouts",
            ]
            if has_reference:
                kept.append("the reference layout")
            if has_families:
                kept.append("one layout per mesh family")
            reasons.append(
                f"this run used <code>--top-k {top_k_limit}</code>, so the payload "
                f"carries {', '.join(kept[:-1])}, and {kept[-1]}"
            )
        else:
            reasons.append(f"the payload carries {carried:,} of them")
    if shown < carried:
        reasons.append(
            f"the report keeps the reference, the first {_EXPLORER_ROWS_PER_SORT_KEY} "
            "of each sort option (exact without a filter), then Pareto-marked points "
            f"and bound order up to {_MAX_EXPLORER_ROWS_PER_CONC} per concurrency"
        )
    full_list = (
        " The full list is in <code>solve --json</code> without <code>--top-k</code>."
        if total_from_funnel
        else ""
    )
    return (
        f"Plots and lists {shown:,} of {total:,} {scope} ({'; '.join(reasons)})."
        f"{ruled_note}{full_list}"
    )


def render_html(payload: Mapping[str, object]) -> str:
    """Render a self-contained, compact Layer-1 per-concurrency offline HTML report."""
    recommended = _as_mapping(payload.get("recommended")) or None
    candidates = [_as_mapping(item) for item in _as_sequence(payload.get("candidates"))]
    scenario = _as_mapping(payload.get("scenario"))
    accept_len = max(1.0, _as_float(scenario.get("accept_len"), 1.0))
    warnings = [str(warning) for warning in _as_sequence(payload.get("warnings"))]
    model = _display_model_label(payload.get("model"))
    has_slo = (
        scenario.get("ttft_slo_ms") is not None
        or scenario.get("tpot_slo_ms") is not None
    )
    chip = str(payload.get("chip") or "TPU")
    report_chip = _payload_chip_spec(payload)
    chips_per_host = (
        report_chip.chips_per_host
        if report_chip is not None
        else CHIPS["tpu7x"].chips_per_host
    )
    has_moe = _detect_has_moe(payload, candidates)
    default_limit_gb = _derive_default_hbm_limit_gb(payload)

    search_map = _as_mapping(payload.get("search"))
    is_hbm_oom = (
        recommended is None
        and bool(search_map)
        and _as_int(search_map.get("hbm_fit"), -1) == 0
    )
    infeasible_msg = (
        "No candidate layout fit HBM memory constraints; "
        "recommended layout is infeasible."
        if is_hbm_oom
        else "No candidate layout met latency SLOs; recommended layout is infeasible."
    )
    top_verdict = recommended.get("slo_verdict") if recommended is not None else None
    rec_infeasible = recommended is None or (
        top_verdict == SLO_VERDICT_RULED_OUT
        if top_verdict is not None
        else not bool(recommended.get("slo_met", True))
    )
    if rec_infeasible and infeasible_msg not in warnings:
        warnings.append(infeasible_msg)

    all_points, ref_key = _collect_all_points(payload)
    is_disagg = any(bool(_as_mapping(pt.get("prefill_pool"))) for pt in all_points)
    concurrencies = sorted({_as_int(pt.get("concurrency")) for pt in all_points})
    surviving_concs = sorted(
        {
            _as_int(pt.get("concurrency"))
            for pt in all_points
            if _point_verdict(pt) != SLO_VERDICT_RULED_OUT
        }
    )
    default_conc: int | None = (
        surviving_concs[-1]
        if surviving_concs
        else (concurrencies[-1] if concurrencies else None)
    )
    default_points = (
        [pt for pt in all_points if _as_int(pt.get("concurrency")) == default_conc]
        if default_conc is not None
        else list(all_points)
    )

    compact_rows = _compact_explorer_rows(
        all_points,
        ref_key=ref_key,
        accept_len=accept_len,
        has_moe=has_moe,
        chip=report_chip,
        default_limit_gb=default_limit_gb,
    )
    default_rows_idx = [
        idx
        for idx, row in enumerate(compact_rows)
        if default_conc is None or _as_int(row.get("c")) == default_conc
    ]
    default_rows = [compact_rows[idx] for idx in default_rows_idx]
    shown_count = sum(
        1 for row in compact_rows if row.get("v") != SLO_VERDICT_RULED_OUT
    )
    carried_count = sum(
        1 for pt in all_points if _point_verdict(pt) != SLO_VERDICT_RULED_OUT
    )
    funnel_total = _funnel_last_stage_count(search_map, has_slo=has_slo)
    not_ruled_out_total = (
        max(shown_count, funnel_total) if funnel_total is not None else None
    )
    listed_total = carried_count if not_ruled_out_total is None else not_ruled_out_total
    raw_top_k_limit = payload.get("top_k_limit")
    scope_note = _explorer_scope_note(
        shown_count,
        listed_total,
        carried=carried_count,
        total_from_funnel=not_ruled_out_total is not None,
        top_k_limit=raw_top_k_limit if isinstance(raw_top_k_limit, int) else None,
        ruled_out_shown=len(compact_rows) - shown_count,
        has_reference=ref_key is not None,
        has_families=bool(_as_mapping(payload.get("bound_families_by_concurrency"))),
    )
    count_label = _shown_of_total(shown_count, listed_total)
    if has_slo:
        count_label += " not ruled out"
    legend_text = _tradeoff_legend_text(
        has_slo=has_slo, has_reference=ref_key is not None
    )
    selected_local_idx = 0
    selected_global_idx = default_rows_idx[0] if default_rows_idx else -1

    # The script selects the first embedded row of the default concurrency, so
    # the first paint picks the same layout with the same sort key.
    selected_point: Mapping[str, object] | None = (
        min(
            default_points,
            key=lambda pt: _explorer_sort_keys(pt, accept_len=accept_len)[0],
        )
        if default_points
        else None
    )

    decode_floor = (
        _as_mapping(selected_point.get("decode_floor"))
        if selected_point is not None
        else None
    )
    selected_pools: list[Mapping[str, object]] = []
    if selected_point is not None:
        for key in ("prefill_pool", "decode_pool"):
            pool_m = _as_mapping(selected_point.get(key))
            if pool_m:
                selected_pools.append(pool_m)
    selected_label = (
        f"Selected: {_explorer_row_label(default_rows[0])} "
        f"(c={_as_int(default_rows[0].get('c'))})"
        if default_rows
        else ""
    )

    raw_families = _as_mapping(payload.get("bound_families_by_concurrency"))
    has_families = bool(raw_families)
    compact_families = (
        _compact_bound_families(raw_families, has_moe=has_moe, chip=report_chip)
        if has_families
        else {}
    )
    ttft_slo_opt = _as_opt_float(scenario.get("ttft_slo_ms"))
    tpot_slo_opt = _as_opt_float(scenario.get("tpot_slo_ms"))
    prefill_pool_recs = _compact_pool_records(
        _as_sequence(payload.get("prefill_pools")),
        has_moe=has_moe,
        chip=report_chip,
        ttft_slo_ms=ttft_slo_opt,
        tpot_slo_ms=tpot_slo_opt,
    )
    decode_pool_recs = _compact_pool_records(
        _as_sequence(payload.get("decode_pools")),
        has_moe=has_moe,
        chip=report_chip,
        ttft_slo_ms=ttft_slo_opt,
        tpot_slo_ms=tpot_slo_opt,
    )
    prefill_pool_rows = _select_pool_rows(prefill_pool_recs, floor_key="btt")
    decode_pool_rows = _select_pool_rows(decode_pool_recs, floor_key="bt")
    has_pools = bool(prefill_pool_recs or decode_pool_recs)

    embedded_payload: dict[str, object] = {
        key: val
        for key, val in payload.items()
        if key
        not in (
            "candidates",
            "frontier",
            "top_k",
            "bound_families_by_concurrency",
            "concurrency_top_k",
            "prefill_pools",
            "decode_pools",
            "ruled_out_candidates",
        )
    }
    embedded_payload["model"] = model
    if has_families:
        embedded_payload["bound_families_by_concurrency"] = compact_families
    embedded_payload["explorer"] = {
        "rows": compact_rows,
        "prefill_pools": prefill_pool_rows,
        "decode_pools": decode_pool_rows,
        "legend": legend_text,
        "is_disagg": is_disagg,
        "has_slo": has_slo,
        "default_conc": default_conc,
        "default_selected_idx": selected_global_idx,
        "default_hbm_limit_gb": round(default_limit_gb, _HBM_GB_DIGITS),
    }
    embedded_json = json.dumps(
        embedded_payload, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")

    tradeoff_view_note = _tradeoff_view_note(
        [compact_rows[idx] for idx in default_rows_idx],
        conc=default_conc,
        mode="est",
    )
    tradeoff_svg = _render_tradeoff_svg(
        [(idx, compact_rows[idx]) for idx in default_rows_idx],
        selected_idx=selected_global_idx,
        mode="est",
        has_slo=has_slo,
    )
    limits_svg = _render_limits_svg(decode_floor)
    memory_svg = _render_memory_svg(selected_pools, default_limit_gb)

    prov_map = _as_mapping(payload.get("provenance"))
    is_calibrated = bool(prov_map.get("calibrated", False))
    mode_badge = (
        '<span class="badge" style="background:#fef3c7;color:#92400e;">'
        "Calibrated Estimates Active</span>"
        if is_calibrated
        else '<span class="badge" style="background:#dcfce7;color:#166534;">'
        "Pure L1 Physics Bounds</span>"
    )
    scen_name = _escape(scenario.get("name", "custom"))
    scen_isl = _as_int(scenario.get("isl"))
    scen_osl = _as_int(scenario.get("osl"))
    scen_hit = _as_float(scenario.get("prefix_hit_rate"))
    meta_suffix = _render_header_meta_suffix(payload, chips_per_host=chips_per_host)
    warnings_html = _render_warnings_html(warnings)
    envelope_html = _render_physical_envelope_card(
        payload,
        all_points,
        recommended=recommended,
        rec_infeasible=rec_infeasible,
        accept_len=accept_len,
        default_limit_gb=default_limit_gb,
        not_ruled_out_total=not_ruled_out_total,
    )
    sh_fit_html = _render_single_host_fit_card(payload, has_moe=has_moe)

    conc_buttons = [
        f'<button type="button" '
        f'class="pill-btn{" active" if c == default_conc else ""}" '
        f'data-conc-btn="{c}">c={c}</button>'
        for c in concurrencies
    ]
    all_active = " active" if default_conc is None else ""
    conc_buttons.append(
        f'<button type="button" class="pill-btn{all_active}" '
        'data-conc-btn="all">All</button>'
    )

    tab_buttons = [
        '<button type="button" class="pill-btn active" data-tab-btn="candidates">'
        f"{'Paired P/D Candidates' if is_disagg else 'Layouts'} "
        f"({count_label})</button>"
    ]
    if prefill_pool_recs:
        tab_buttons.append(
            '<button type="button" class="pill-btn" data-tab-btn="prefill">'
            "Prefill Pools "
            f"({_shown_of_total(len(prefill_pool_rows), len(prefill_pool_recs))})"
            "</button>"
        )
    if decode_pool_recs:
        tab_buttons.append(
            '<button type="button" class="pill-btn" data-tab-btn="decode">'
            "Decode Pools "
            f"({_shown_of_total(len(decode_pool_rows), len(decode_pool_recs))})"
            "</button>"
        )
    if has_families:
        fam_total = sum(len(v) for v in compact_families.values())
        tab_buttons.append(
            '<button type="button" class="pill-btn" data-tab-btn="families">'
            f"By Mesh Family ({fam_total})</button>"
        )

    pool_header_html = (
        "          <th>Prefill Pool</th>\n          <th>Decode Pool</th>"
        if is_disagg
        else "          <th>Pool</th>"
    )
    moe_note = (
        "MoE pools show Attention axes (<code>Attn: ...</code>) and Expert axes "
        "(<code>MoE: EP...</code> whole-expert sharding vs. "
        "<code>MoE: TP...</code> width-sliced experts). "
        if has_moe
        else ""
    )
    ref_report = _as_mapping(payload.get("reference"))
    ref_missing_html = (
        f'    <p class="section-note"># reference layout '
        f"{_escape(str(ref_report.get('layout', '')))}: not enumerated "
        f"({_escape(str(ref_report.get('rejected_by', '')))})</p>"
        if (ref_report and not bool(ref_report.get("found")))
        else ""
    )

    pool_table_head = (
        "<th>SLO Verdict</th><th>TPOT Floor (ms)</th><th>TTFT Floor (ms)</th>"
        "<th>Est. Tput (tok/s/chip) [est]</th><th>HBM</th><th>Binding</th>"
    )
    decode_pool_note_html = (
        '        <p class="section-note">Each concurrency lists its first '
        f"{_POOL_ROWS_PER_CONC} decode pools by TPOT floor. A decode pool's floor "
        "is its hard step floor at the batch that concurrency puts on it, so it "
        "can be higher than the TPOT Bound of a pair that uses the pool.</p>"
    )
    prefill_pool_note_html = (
        '        <p class="section-note">Each concurrency lists its first '
        f"{_POOL_ROWS_PER_CONC} prefill pools by TTFT floor. A prefill pool's "
        "TTFT floor is the same at every concurrency.</p>"
    )
    pool_panels_html: list[str] = []
    if has_pools:
        pool_panels_html.append('    <div id="pool-tables">')
        if prefill_pool_recs:
            pool_panels_html.extend(
                [
                    '      <div id="panel-prefill" class="hidden-panel table-wrap">',
                    prefill_pool_note_html,
                    "        <table><thead><tr>",
                    "          <th>Row</th><th>Concurrency</th><th>Prefill Pool</th>"
                    f"{pool_table_head}",
                    '        </tr></thead><tbody id="prefill-pools-tbody">',
                    "        "
                    + _render_static_pool_rows(
                        prefill_pool_rows, default_conc=default_conc, has_slo=has_slo
                    ),
                    "        </tbody></table>",
                    "      </div>",
                ]
            )
        if decode_pool_recs:
            pool_panels_html.extend(
                [
                    '      <div id="panel-decode" class="hidden-panel table-wrap">',
                    decode_pool_note_html,
                    "        <table><thead><tr>",
                    "          <th>Row</th><th>Concurrency</th><th>Decode Pool</th>"
                    f"{pool_table_head}",
                    '        </tr></thead><tbody id="decode-pools-tbody">',
                    "        "
                    + _render_static_pool_rows(
                        decode_pool_rows, default_conc=default_conc, has_slo=has_slo
                    ),
                    "        </tbody></table>",
                    "      </div>",
                ]
            )
        pool_panels_html.append("    </div>")

    family_panel_html: list[str] = []
    if has_families:
        family_panel_html.extend(
            [
                '    <div id="bound-families-table" class="hidden-panel table-wrap">',
                "      <table><thead><tr>",
                "        <th>Concurrency</th><th>Rank</th><th>Mesh Family</th>"
                "<th>Min TPOT Bound (ms) [bound]</th>"
                "<th>Min TTFT Bound (ms) [bound]</th>"
                "<th>Envelope TPOT / TTFT (ms)</th><th>Binding</th><th>Members</th>",
                '      </tr></thead><tbody id="families-tbody">',
                f"        {_render_static_family_rows(compact_families, default_conc)}",
                "      </tbody></table>",
                "    </div>",
            ]
        )

    static_explorer_body = _render_static_explorer_rows(
        default_rows,
        is_disagg=is_disagg,
        selected_idx=selected_local_idx,
        has_slo=has_slo,
    )
    step_bound_note = (
        f"TPOT Bound = Hard Floor / {accept_len:g} accepted tokens per step."
        if accept_len > 1.0
        else "TPOT Bound = Hard Floor."
    )
    toggle_style = (
        "" if len(default_rows) > _DEFAULT_VISIBLE_ROWS else ' style="display:none;"'
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
        "    <h1>TPU Roofline Layer-1 Explorer Report</h1>",
        '    <div class="meta">',
        f'      <span class="badge">{_escape(chip)}</span>{mode_badge}',
        f"      <strong>Model:</strong> {_escape(model)} &bull; ",
        f"      <strong>Scenario:</strong> {scen_name} "
        f"(ISL={scen_isl:,}, OSL={scen_osl:,}, hit_rate={scen_hit:.2f}){meta_suffix}",
        "    </div>",
        "  </header>",
        f"  {warnings_html}",
        envelope_html,
        sh_fit_html,
        *_render_funnel_section(payload),
        '  <section id="concurrency-controls" class="card">',
        '    <div class="controls-row">',
        '      <div class="pill-group">',
        '        <span style="font-size:12.5px;font-weight:700;color:#334155;">'
        "Concurrency:</span>",
        f"        {' '.join(conc_buttons)}",
        "      </div>",
        '      <span class="section-note">Per-concurrency L1 view &middot; '
        "Click any plot point or table row to inspect Step/HBM breakdown</span>",
        "    </div>",
        "  </section>",
        '  <section id="frontier-chart" class="card">',
        '    <div class="chart-header">',
        "      <h2>Per-Concurrency Trade-off "
        "(Throughput per Chip vs. Interactivity)</h2>",
        '      <div class="pill-group">',
        '        <button type="button" class="pill-btn active" data-plot-btn="est">'
        "Estimates (est_*)</button>",
        '        <button type="button" class="pill-btn" data-plot-btn="bound">'
        "Physical Bounds (bound_*)</button>",
        "      </div>",
        "    </div>",
        f'    <p class="section-note">{scope_note} '
        "<code>est_*</code> envelopes are uncalibrated reference guides and never "
        "prune layouts.</p>",
        f'    <p id="tradeoff-view-note" class="section-note">'
        f"{_escape(tradeoff_view_note)}</p>",
        f'    <div id="tradeoff-svg-wrap" class="chart-container">{tradeoff_svg}</div>',
        '    <div id="tradeoff-legend" class="chart-legend">'
        f"{_escape(legend_text['est'])}</div>",
        "  </section>",
        '  <section id="candidate-explorer" class="card">',
        '    <div class="chart-header">',
        "      <h2>Per-Concurrency Candidate Explorer</h2>",
        '      <div class="pill-group">',
        f"        {' '.join(tab_buttons)}",
        "      </div>",
        "    </div>",
        '    <div class="controls-row" style="margin-top:10px;">',
        '      <div class="pill-group" id="explorer-row-controls">',
        '        <label for="explorer-sort" style="font-size:12px;font-weight:600;'
        'color:#475569;">Sort by:</label>',
        '        <select id="explorer-sort" class="sort-select">',
        '          <option value="bt">TPOT Bound (asc) [bound]</option>',
        '          <option value="fc">Tput Ceiling (desc) [bound]</option>',
        '          <option value="et">Est. Tput (desc) [est]</option>',
        '          <option value="ei">Est. Interactivity (desc) [est]</option>',
        '          <option value="hbm">HBM Footprint (asc) [bound]</option>',
        "        </select>",
        '        <input id="explorer-filter" class="filter-input" type="search" '
        'placeholder="Filter layout, topology, binding, verdict..."/>',
        "      </div>",
        f'      <button type="button" id="explorer-toggle-all" class="pill-btn"'
        f"{toggle_style}>Show All ({len(default_rows)})</button>",
        "    </div>",
        f'    <p class="section-note">{moe_note}'
        "Columns marked <code>[bound]</code> are physical lower limits / ceilings; "
        "<code>[est]</code> are uncalibrated serial-phase analytical estimates. "
        "Row is the row number in the current order; when a tab lists a subset, "
        f"rows after the first {_DEFAULT_VISIBLE_ROWS} are not ranks.</p>",
        '    <div id="panel-candidates" class="table-wrap">',
        "      <table>",
        "        <thead>",
        "          <tr>",
        "            <th>Row</th>",
        "            <th>Chips</th>",
        pool_header_html,
        "            <th>SLO Verdict</th>",
        "            <th>TPOT Bound (ms) [bound]</th>",
        "            <th>TTFT Bound (ms) [bound]</th>",
        "            <th>Tput Ceiling (tok/s/chip) [bound]</th>",
        "            <th>Est. Int (tok/s) [est]</th>",
        "            <th>Est. Tput (tok/s/chip) [est]</th>",
        "            <th>Est. TPOT (ms) [est]</th>",
        "            <th>KV Cap / Prefix Hit Rate</th>",
        "            <th>Binding</th>",
        "          </tr>",
        "        </thead>",
        '        <tbody id="explorer-tbody">',
        f"        {static_explorer_body}",
        "        </tbody>",
        "      </table>",
        "    </div>",
        *pool_panels_html,
        *family_panel_html,
        ref_missing_html,
        "  </section>",
        '  <section id="limits-chart" class="card">',
        "    <h2>Decode Step Resource Breakdown</h2>",
        f'    <p id="selected-layout-limits-meta" class="section-note">'
        f"{_escape(selected_label)}</p>",
        '    <p class="section-note">'
        "One decode step with the fewest resident sequences: per-resource lower "
        "bounds (<code>bound_*</code>) above the dashed line, serial estimate "
        "overhead (<code>est_*</code>) below it. The table's "
        "<code>Binding</code> column reflects the operating batch at the "
        "selected concurrency, so it can name a different resource than this "
        f"minimum-batch floor. {step_bound_note}</p>",
        f'    <div id="limits-svg-wrap" class="chart-container">{limits_svg}</div>',
        "  </section>",
        '  <section id="memory-chart" class="card">',
        "    <h2>HBM Memory Footprint vs. Capacity</h2>",
        f'    <p id="selected-layout-memory-meta" class="section-note">'
        f"{_escape(selected_label)}</p>",
        f'    <div id="memory-svg-wrap" class="chart-container">{memory_svg}</div>',
        f"    {_MEMORY_LEGEND_HTML}",
        "  </section>",
        *_render_assumptions_card(payload, report_chip),
        "  <footer>",
        "    Analytical Layer-1 roofline report generated by "
        "<code>vllm_torchtpu.roofline</code> &bull; "
        "Bounds (<code>bound_*</code>, <code>floor_throughput_per_chip</code>) are "
        "physical limits; estimates (<code>est_*</code>) are uncalibrated model "
        "estimates",
        "  </footer>",
        "</div>",
        '<script type="application/json" id="roofline-data">',
        f"{embedded_json}",
        "</script>",
        f"<script>{JS_INTERACTIVE}</script>",
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
