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
"""Interactive 3-pane HTML workbench for megakernel fusion-scope analysis."""

from __future__ import annotations

import html

from .assets import load_asset
from .constants import VALID_MESH_AXIS_VALUES
from .workbench_engine import render_workbench_engine_js
from .workbench_phase import (
    render_phase_extension_js,
    render_phase_stage_tab_button,
)


def _esc_attr(value: object) -> str:
    """Escape a scalar value for safe embedding in HTML text or attributes."""
    return "" if value is None else html.escape(str(value), quote=True)


def _rknob(
    key: str,
    title: str,
    init_lbl: str,
    bounds: tuple[float, float, float, float],
    fn: str = "wbOnControl()",
) -> str:
    """Render a labeled range-slider control row."""
    lo, hi, step, val = bounds
    return (
        f'<div class="wb-knob-row"><div class="wb-knob-head"><span>{title}</span>'
        f'<span class="tabular" id="lbl-{key}">{init_lbl}</span></div>'
        f'<input id="ctl-{key}" type="range" min="{lo}" max="{hi}" '
        f'step="{step}" value="{val}" oninput="{fn}"/></div>'
    )


def _sknob(
    key: str,
    title: str,
    init_lbl: str,
    options: tuple[tuple[str, str], ...],
    fn: str = "wbOnControl()",
) -> str:
    """Render a labeled select-dropdown control row."""
    opts = "".join(f'<option value="{val}">{lbl}</option>' for val, lbl in options)
    return (
        f'<div class="wb-knob-row"><div class="wb-knob-head"><span>{title}</span>'
        f'<span class="tabular" id="lbl-{key}">{init_lbl}</span></div>'
        f'<select id="ctl-{key}" onchange="{fn}">{opts}</select></div>'
    )


def _mesh_axis_options() -> tuple[tuple[str, str], ...]:
    """Return physical power-of-two sharding options for mesh dropdowns."""
    return tuple((str(v), str(v)) for v in VALID_MESH_AXIS_VALUES)


def render_workbench_css() -> str:
    """Return CSS rules for the 3-pane viewport workbench and audit switcher."""
    return load_asset("workbench.css")


def render_workbench_body(
    *,
    model_name: str,
    chip_name: str,
    layout_label: str,
    mode_source: str,
    vmem_bounds_mib: tuple[float, float] = (62.0, 62.0),
) -> str:
    """Render the 3-pane Fusion Workbench HTML shell."""
    tabs_html = render_phase_stage_tab_button()
    vmem_init, vmem_max = vmem_bounds_mib
    m_e, c_e = _esc_attr(model_name), _esc_attr(chip_name)
    l_e, s_e = _esc_attr(layout_label), _esc_attr(mode_source)
    phases = tuple((p, p) for p in ("decode", "verify", "prefill", "mixed"))
    dma_o, al_o = (("2", "2 slots"), ("3", "3 slots")), (("1", "on"), ("0", "off"))
    hw_fn = "wbOnHwControl()"
    mesh_opts = _mesh_axis_options()
    mesh_fn = "wbOnMeshChange()"
    mesh_grid = (
        '<div class="wb-mesh-grid">'
        + _sknob("tp", "TP", "8", mesh_opts, mesh_fn)
        + _sknob("ep", "EP", "1", mesh_opts, mesh_fn)
        + _sknob("dp", "DP", "1", mesh_opts, mesh_fn)
        + _sknob("pp", "PP", "1", mesh_opts, mesh_fn)
        + _sknob("dcp", "DCP", "1", mesh_opts, mesh_fn)
        + _sknob("pcp", "PCP", "1", mesh_opts, mesh_fn)
        + "</div>"
    )
    pair_wl = _sknob("phase", "Phase", "decode", phases, "wbOnPhaseChange()") + _sknob(
        "layout", "Parallelism", l_e, (), "wbOnLayoutPresetChange()"
    )
    mesh_drawer = (
        '<details class="wb-eta-details" id="wb-mesh-details">'
        "<summary>Custom 6-Axis Mesh Override (TP / EP / DP / PP / DCP / PCP)"
        f"</summary>{mesh_grid}</details>"
    )
    knobs_wl = (
        f'<div class="wb-knob-pair">{pair_wl}</div>'
        + _rknob("batch", "Batch Size (B)", "16", (1, 1024, 1, 16))
        + _rknob("seq-tokens", "Tokens/Seq", "1", (1, 8192, 1, 1), "wbOnSeqChange()")
        + _rknob("context", "Context (S)", "4.0K", (512, 1048576, 512, 4096))
        + mesh_drawer
    )
    dr_btns = (
        '<div class="wb-knob-head"><span>Compiler / Dispatch Regime</span>'
        '<span class="tabular" id="lbl-dispatch-regime">compiled_xla</span></div>'
        '<div class="wb-dr-group">'
        '<button class="wb-mini-btn" id="btn-dr-zero" type="button" '
        "onclick=\"wbSetDispatchRegime('zero_floor')\">Zero-Gap Floor</button>"
        '<button class="wb-mini-btn active-a" id="btn-dr-xla" type="button" '
        "onclick=\"wbSetDispatchRegime('compiled_xla')\">Compiled XLA</button>"
        '<button class="wb-mini-btn" id="btn-dr-eager" type="button" '
        "onclick=\"wbSetDispatchRegime('unfused_eager')\">Exposed Dispatch</button>"
        "</div>"
    )
    eta_block = (
        '<details class="wb-eta-details" id="wb-eta-details">'
        "<summary>Achievable Efficiency (&eta;)</summary>"
        + _rknob(
            "eta-dense", "Dense Compute eta", "0.72", (0.10, 1.0, 0.02, 0.72), hw_fn
        )
        + _rknob("eta-moe", "Grouped MoE eta", "0.50", (0.10, 1.0, 0.02, 0.50), hw_fn)
        + _rknob("eta-hbm", "HBM Stream eta", "0.82", (0.10, 1.0, 0.02, 0.82), hw_fn)
        + _rknob(
            "eta-ici", "ICI Collective eta", "0.75", (0.10, 1.0, 0.02, 0.75), hw_fn
        )
        + "</details>"
    )
    pair_hw = _sknob("dma", "DMA Slots", "2 slots", dma_o, hw_fn) + _sknob(
        "alias", "Scratch Alias", "enabled", al_o, hw_fn
    )
    adv_hw = (
        '<details class="wb-eta-details" id="wb-adv-hw-details">'
        "<summary>Advanced Priors (Gap, Hop, Ring Slots, Aliasing)</summary>"
        + _rknob("launch-gap", "Launch Gap", "1.00 us", (0, 12, 0.01, 1.0), hw_fn)
        + _rknob("coll-hop", "Collective Hop", "1.00 us", (0.5, 8.0, 0.001, 1.0), hw_fn)
        + f'<div class="wb-knob-pair">{pair_hw}</div>'
        + "</details>"
    )
    knobs_hw = (
        dr_btns
        + _rknob(
            "vmem",
            f"VMEM Budget (max {vmem_max:.0f} MiB)",
            f"{vmem_init:.1f} MiB",
            (2.0, vmem_max, 0.5, vmem_init),
            hw_fn,
        )
        + adv_hw
        + eta_block
    )
    sc_opts = "".join(
        f'<option value="{s}">{s}</option>'
        for s in ("whole_step", "layer_group", "single_layer", "per_op")
    )
    kpis = "".join(
        f'<div class="wb-pill-kpi">{label}: '
        f'<strong class="tabular" id="wb-kpi-{k}">-</strong></div>'
        for k, label in (
            ("scope", "SCOPE"),
            ("floor", "FLOOR"),
            ("tmax", "IDEAL TMAX"),
            ("tps", "TPS"),
        )
    )
    return f"""  <header class="wb-topbar" id="wb-topbar"><div class="wb-brand">
  <h1>Megakernel Fusion Workbench &middot; {m_e}</h1><div class="wb-sub">Chip:
  <strong>{c_e}</strong> &middot; Mesh: <strong id="wb-active-layout">{l_e}</strong>
  ({s_e})</div></div><div class="wb-verdict-wrap">
  <span class="wb-verdict-pill" id="wb-verdict-pill">BUILD WHOLE_STEP</span>
  <div class="wb-verdict-meta tabular" id="wb-verdict-summary">-</div></div>
  <div class="wb-top-kpis">{kpis}</div><div class="wb-actions">
  <button class="wb-tab-btn active" id="btn-view-workbench" type="button"
  onclick="wbSetView('workbench')">Fusion Workbench</button><button class="wb-tab-btn"
  id="btn-view-audit" type="button" onclick="wbSetView('audit')">Audit Report &amp;
  Tables</button><button class="theme-btn" id="theme-toggle" onclick="toggleTheme()"
  type="button" aria-label="Toggle theme"><span>&#9680;</span> Theme</button></div>
  </header><div id="wb-invalid-banner" class="wb-invalid-banner view-hidden"
  role="alert"></div><main class="wb-grid" id="view-workbench"><aside class="wb-rail"
  id="wb-left-rail"><div class="wb-cuj-hint" id="wb-cuj-hint"><b>Quick Start:</b>
  1. Click a row to pin <b>[A]</b> &middot; 2. Click <b>B</b> to compare &middot;
  3. Open right-pane dropdowns for equations.</div>
  <div class="wb-sec-title">1. Request Regimes ([A] vs [B])</div>
  <div style="display:flex;gap:5px;margin-bottom:4px;"><button class="wb-mini-btn"
  id="btn-swap-ab" type="button" onclick="wbSwapAB()">Swap [A]/[B]</button><button
  class="wb-mini-btn active-b" id="btn-vs-perop" type="button"
  onclick="wbClearB()">vs per_op</button><button class="wb-mini-btn"
  id="btn-reset-all" type="button" onclick="wbResetSliders()">Reset</button></div>
  <div class="kpi-sub" id="wb-ab-status" style="margin-bottom:5px;font-size:10px"></div>
  <div class="wb-regime-list" id="wb-regime-list"></div>
  <div class="wb-sec-title">2. On-Chip VMEM &amp; Runtime Priors</div>
  <div class="wb-knob-group">{knobs_hw}</div>
  <div class="wb-sec-title">3. Live Workload &amp; Mesh Knobs</div>
  <div class="wb-knob-group">{knobs_wl}</div><div id="wb-extra-rail-slot"></div></aside>
  <section class="wb-stage" id="wb-center-stage"><div id="wb-stage-tabs"
  style="display:flex;gap:8px;flex-wrap:wrap;"><button class="wb-tab-btn active"
  id="btn-stage-arena" type="button" onclick="wbSetStageTab('arena')">VMEM Floorplan
  &amp; Waterfall</button>{tabs_html}</div><div id="wb-stage-arena"
  style="display:flex;flex-direction:column;gap:12px;"><div class="wb-panel">
  <div class="wb-panel-head"><h2>On-Chip VMEM Spatial Floorplan &amp; Lifetime Aliasing
  Arena</h2><select class="wb-inline-sel" id="ctl-arena-scope"
  onchange="wbOnArenaScopeChange()">{sc_opts}</select></div><div
  id="wb-vmem-arena-svg"></div>
  </div><div class="wb-panel"><div class="wb-panel-head"><h2>Shared-Axis 4-Scope Step
  Timeline &amp; Mechanism Envelope</h2><span class="kpi-sub">Click a scope row for
  Proof-Ladder equations</span></div><div id="wb-scope-timeline-svg"></div></div>
  <div class="wb-panel"><div class="wb-panel-head"><h2 id="wb-waterfall-title">Signed
  Attribution</h2><span class="kpi-sub" id="wb-waterfall-summary"></span></div>
  <div id="wb-waterfall-svg"></div></div></div><div id="wb-extra-stage-slot"></div>
  </section><aside class="wb-inspector" id="wb-right-inspector">
  <div class="wb-sec-title">Proof-Ladder Equation &amp; Provenance</div>
  <div id="wb-deriv-drawer"></div></aside></main>"""


def render_workbench_script(model_name: str) -> str:
    """Return the ES2022 workbench script of one model's report.

    It combines the physics engine and the SVG views. The model name selects
    the deployment role layouts of the phase map.
    """
    ui_js = load_asset("workbench.js")
    return (
        f"{render_workbench_engine_js()}\n"
        f"{ui_js}\n"
        f"{render_phase_extension_js(model_name)}\n"
        "  window.wbRenderAll();\n"
    )
