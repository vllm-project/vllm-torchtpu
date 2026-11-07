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

from .workbench_engine import render_workbench_engine_js


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


def render_workbench_css() -> str:
    """Return CSS rules for the 3-pane viewport workbench and audit switcher."""
    return """  .wb-mode-active{overflow:hidden;height:100vh;padding:0}.wb-topbar{
  display:flex;justify-content:space-between;align-items:center;gap:12px;
  padding:8px 16px;background:var(--card);min-height:50px;box-sizing:border-box;
  border-bottom:1px solid var(--border)}.wb-brand h1{font-size:16px;margin:0;
  font-weight:700}.wb-sub{font-size:11px;color:var(--muted)}.wb-top-kpis,.wb-actions{
  display:flex;gap:8px;align-items:center}.wb-pill-kpi{font-size:11px;
  background:var(--card-inner);border-radius:6px;padding:4px 9px;
  border:1px solid var(--border-subtle)}.wb-pill-kpi strong{font-size:12px;
  margin-left:4px}.wb-tab-btn{background:var(--card-inner);color:var(--text);
  border:1px solid var(--border);border-radius:6px;padding:5px 10px;font-size:12px;
  font-weight:600;cursor:pointer}.wb-tab-btn.active{background:var(--accent);
  color:#FFF}.wb-grid{display:grid;grid-template-columns:294px minmax(0,1fr) 340px;
  height:calc(100vh - 51px);overflow:hidden}.wb-rail,.wb-inspector{overflow-y:auto;
  background:var(--card);padding:10px 12px;box-sizing:border-box}.wb-rail{
  border-right:1px solid var(--border)}.wb-inspector{
  border-left:1px solid var(--border)}.wb-stage{overflow-y:auto;padding:12px 16px;
  background:var(--bg);display:flex;flex-direction:column;gap:12px;
  box-sizing:border-box}.wb-rail,.wb-stage,.wb-inspector,.wb-regime-list,.wb-formula{
  scrollbar-width:thin;scrollbar-color:var(--border) transparent}.wb-sec-title{
  font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);margin:0 0 5px}.wb-regime-list{display:flex;flex-direction:column;
  gap:4px;margin-bottom:8px;max-height:164px;overflow-y:auto;padding-right:4px}
  .wb-regime-item{display:flex;justify-content:space-between;gap:4px;cursor:pointer;
  align-items:center;padding:5px 7px;border-radius:6px;font-size:11px;
  border:1px solid var(--border-subtle);background:var(--card-inner)}
  .wb-regime-item.pin-a{border-color:#009E73;box-shadow:inset 3px 0 0 #009E73}
  .wb-regime-item.pin-b{border-color:#56B4E9;box-shadow:inset 3px 0 0 #56B4E9}
  .wb-mini-btn{background:var(--card);border:1px solid var(--border);cursor:pointer;
  color:var(--text);border-radius:4px;padding:2px 6px;font-size:10px;font-weight:700}
  .wb-mini-btn.active-a{background:#009E73;color:#FFF;border-color:#009E73}
  .wb-mini-btn.active-b{background:#56B4E9;color:#FFF;border-color:#56B4E9}
  .wb-knob-group{display:flex;flex-direction:column;gap:5px;padding:8px;
  margin-bottom:8px;border-radius:6px;background:var(--card-inner);
  border:1px solid var(--border-subtle)}.wb-knob-pair{display:grid;gap:6px;
  grid-template-columns:1fr 1fr}.wb-knob-row{display:flex;font-size:11px;
  flex-direction:column;gap:2px}.wb-knob-head{display:flex;color:var(--muted);
  justify-content:space-between}.wb-knob-head span:last-child{color:var(--text);
  font-weight:600}.wb-knob-row input[type="range"]{width:100%;margin:0;
  accent-color:var(--accent)}.wb-knob-row select,.wb-inline-sel{font-size:11px;
  background:var(--card);color:var(--text);border:1px solid var(--border);
  border-radius:4px;padding:3px 5px}.wb-panel{background:var(--card);padding:12px 14px;
  border:1px solid var(--border);border-radius:8px;box-shadow:var(--shadow)}
  .wb-panel-head h2{margin:0;font-size:14px}.wb-panel-head{display:flex;gap:8px;
  justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap}
  .wb-deriv-box{background:var(--card-inner);border-radius:6px;padding:9px 10px;
  border:1px solid var(--border-subtle);margin-bottom:8px;font-size:12px}
  .wb-formula{font-family:monospace;font-size:11px;padding:6px 8px;margin:5px 0;
  background:var(--code-bg);border-radius:4px;overflow-x:auto;white-space:pre-wrap;
  overflow-wrap:anywhere;word-break:break-word}.wb-clickable{cursor:pointer}
  .wb-clickable:hover{opacity:.88}.view-hidden{display:none !important}
  .audit-scroll-wrap{padding:24px}"""


def render_workbench_body(
    *,
    model_name: str,
    chip_name: str,
    layout_label: str,
    mode_source: str,
    vmem_bounds_mib: tuple[float, float] = (62.0, 62.0),
) -> str:
    """Render the 3-pane Fusion Workbench HTML shell."""
    tabs_html = ""
    vmem_init, vmem_max = vmem_bounds_mib
    m_e, c_e = _esc_attr(model_name), _esc_attr(chip_name)
    l_e, s_e = _esc_attr(layout_label), _esc_attr(mode_source)
    phases = tuple((p, p) for p in ("decode", "verify", "prefill", "mixed"))
    dma_o, al_o = (("2", "2 slots"), ("3", "3 slots")), (("1", "on"), ("0", "off"))
    hw_fn = "wbOnHwControl()"
    pair_wl = _sknob("phase", "Phase", "decode", phases, "wbOnPhaseChange()") + _sknob(
        "layout", "Parallelism", l_e, ()
    )
    knobs_wl = (
        f'<div class="wb-knob-pair">{pair_wl}</div>'
        + _rknob("batch", "Batch Size (B)", "16", (1, 1024, 1, 16))
        + _rknob("seq-tokens", "Tokens/Seq", "1", (1, 8192, 1, 1), "wbOnSeqChange()")
        + _rknob("context", "Context (S)", "4.0K", (512, 131072, 512, 4096))
    )
    pair_hw = _sknob("dma", "DMA Slots", "2 slots", dma_o, hw_fn) + _sknob(
        "alias", "Scratch Alias", "enabled", al_o, hw_fn
    )
    knobs_hw = (
        _rknob(
            "vmem",
            f"VMEM Budget (max {vmem_max:.0f} MiB)",
            f"{vmem_init:.1f} MiB",
            (2.0, vmem_max, 0.5, vmem_init),
            hw_fn,
        )
        + _rknob("launch-gap", "Launch Gap", "0.00 us", (0, 12, 0.01, 0), hw_fn)
        + _rknob("coll-hop", "Collective Hop", "1.00 us", (0.5, 8.0, 0.001, 1.0), hw_fn)
        + f'<div class="wb-knob-pair">{pair_hw}</div>'
    )
    sc_opts = "".join(
        f'<option value="{s}">{s}</option>'
        for s in ("whole_step", "layer_group", "single_layer", "per_op")
    )
    kpis = "".join(
        f'<div class="wb-pill-kpi">{k.upper()}: '
        f'<strong class="tabular" id="wb-kpi-{k}">-</strong></div>'
        for k in ("scope", "floor", "tmax", "tps")
    )
    return f"""  <header class="wb-topbar" id="wb-topbar"><div class="wb-brand">
  <h1>Megakernel Fusion Workbench &middot; {m_e}</h1><div class="wb-sub">Chip:
  <strong>{c_e}</strong> &middot; Mesh: <strong id="wb-active-layout">{l_e}</strong>
  ({s_e})</div></div><div class="wb-top-kpis">{kpis}</div><div class="wb-actions">
  <button class="wb-tab-btn active" id="btn-view-workbench" type="button"
  onclick="wbSetView('workbench')">Fusion Workbench</button><button class="wb-tab-btn"
  id="btn-view-audit" type="button" onclick="wbSetView('audit')">Audit Report &amp;
  Tables</button><button class="theme-btn" id="theme-toggle" onclick="toggleTheme()"
  type="button" aria-label="Toggle theme"><span>&#9680;</span> Theme</button></div>
  </header><main class="wb-grid" id="view-workbench"><aside class="wb-rail"
  id="wb-left-rail"><div class="wb-sec-title">1. Request Regimes ([A] vs [B])</div>
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
  onchange="wbRenderAll()">{sc_opts}</select></div><div id="wb-vmem-arena-svg"></div>
  </div><div class="wb-panel"><div class="wb-panel-head"><h2>Shared-Axis 4-Scope Step
  Timeline &amp; Mechanism Envelope</h2><span class="kpi-sub">Click a scope row for
  Proof-Ladder equations</span></div><div id="wb-scope-timeline-svg"></div></div>
  <div class="wb-panel"><div class="wb-panel-head"><h2 id="wb-waterfall-title">Signed
  Attribution</h2><span class="kpi-sub" id="wb-waterfall-summary"></span></div>
  <div id="wb-waterfall-svg"></div></div></div><div id="wb-extra-stage-slot"></div>
  </section><aside class="wb-inspector" id="wb-right-inspector">
  <div class="wb-sec-title">Proof-Ladder Equation &amp; Provenance</div>
  <div id="wb-deriv-drawer"></div></aside></main>"""


def render_workbench_script() -> str:
    """Return ES2022 workbench script combining physics engine and SVG views."""
    ui_js = r"""
  (function() { const ENG = window.__MK_ADVISOR_ENGINE; if (!ENG) return;
  const { DATA, CHIP, REGIMES, evaluateWorkload } = ENG;
  const el = id => document.getElementById(id);
  const escH = s => String(s == null ? '' : s).replace(/&/g, '&amp;')
  .replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const rSvg = (x, y, w, h, c) => `<rect x="${x.toFixed(1)}" y="${y}" ` +
  `width="${w.toFixed(1)}" height="${h}" fill="${c}"/>`;
  const dBox = (t, tag, body) => `<div class="wb-deriv-box"><div style="display:flex;` +
  `justify-content:space-between;"><strong>${t}</strong><span class="prov-tag ` +
  `prov-tag-${tag}">${tag}</span></div>${body}</div>`;
  const SCOPE_COLORS = { whole_step: '#009E73', layer_group: '#56B4E9',
  single_layer: '#E69F00', per_op: '#CC79A7' };
  const STATE = { pinA: 0, pinB: -1, customDirty: false, hwDirty: false,
  calibrated: false, selectedTopic: { kind: 'scope', scope: 'whole_step',
  sub: 'total' }, calSource: '', calStepTimeUs: null, launchGapHiUs: 8.0,
  denseSlotBytes: 2097152, collHopHiUs: (CHIP.collective_latency_us_bracket ||
  [1.0, 5.13])[1], expertSlotBytes: 2097152, presets: [], resetSeq: 0,
  maxVmemMib: Math.round(((CHIP.vmem_usable_bytes || 65011712) / 1048576) * 10) / 10 };
  function initPresets() {
  if (el('ctl-vmem')) {
    el('ctl-vmem').min = '2'; el('ctl-vmem').max = String(STATE.maxVmemMib);
  }
  const defP = DATA.parallelism || { tp: 8, ep: 1, dp: 1, pp: 1, dcp: 1, pcp: 1 };
  const seen = new Set(), addP = (name, p) => { if (!p || !p.tp) return;
  const key = [p.tp, p.ep || 1, p.dp || 1, p.pp || 1, p.dcp || 1, p.pcp || 1].join(':');
  if (seen.has(key)) return; seen.add(key); STATE.presets.push({ name, tp: p.tp,
  ep: p.ep || 1, dp: p.dp || 1, pp: p.pp || 1, dcp: p.dcp || 1, pcp: p.pcp || 1,
  layerGroups: p.layer_groups || p.layerGroups || [] }); };
  addP('Decode Mesh', defP); addP('Prefill Mesh', DATA.prefill_parallelism);
  addP('Saturated Mesh', DATA.saturation_parallelism);
  for (const r of REGIMES) if (r && r.workload) addP(r.layout || 'Mesh', r.workload);
  if (el('ctl-layout')) el('ctl-layout').innerHTML = STATE.presets.map((p, i) =>
  `<option value="${i}">${escH(p.name)} (TP${p.tp} x EP${p.ep})</option>`).join(''); }
  function runtimeKnobs(gapFallback) {
  const rawV = parseFloat(el('ctl-vmem').value) || DATA.vmem_budget_mib
  || STATE.maxVmemMib;
  const clampedV = Math.min(STATE.maxVmemMib, Math.max(2.0, rawV));
  return { vmemMip: clampedV,
  dmaBuffers: parseInt(el('ctl-dma').value, 10) || DATA.dma_buffers || 2,
  vmemAlias: el('ctl-alias').value === '1',
  launchGapUs: (STATE.customDirty || STATE.hwDirty || STATE.calibrated)
  ? (parseFloat(el('ctl-launch-gap').value) || 0.0) : gapFallback,
  collHopUs: parseFloat(el('ctl-coll-hop').value) || 1.0,
  launchGapHiUs: STATE.launchGapHiUs, collHopHiUs: STATE.collHopHiUs,
  denseSlotBytes: STATE.denseSlotBytes, expertSlotBytes: STATE.expertSlotBytes }; }
  function regimeToWorkload(idx) {
  const reg = REGIMES[idx] || REGIMES[0] || {}, rw = reg.workload || {};
  const ph = rw.phase || reg.phase || 'decode', seqTok = ph === 'decode' ? 1
  : (ph === 'verify' ? ((rw.spec_tokens || 1) + 1) : (rw.chunk_size || rw.isl || 1));
  const gapDef = rw.op_overhead_us !== undefined ? rw.op_overhead_us : 0.0;
  return Object.assign({ label: reg.label || 'Preset', phase: ph,
  batch: rw.batch || reg.active_tokens || 16, seqTokens: seqTok, isl: rw.isl,
  osl: rw.osl, specTokens: rw.spec_tokens,
  context: rw.context || reg.effective_context || 4096, chunkSize: rw.chunk_size,
  prefixHitRate: rw.prefix_hit_rate || 0.0, microbatches: rw.microbatches || 0,
  segmentLayers: rw.segment_layers || 0, tp: rw.tp || 8, ep: rw.ep || 1,
  dp: rw.dp || 1, pp: rw.pp || 1, dcp: rw.dcp || 1, pcp: rw.pcp || 1,
  kvSplit: rw.kv_split || 1, layerGroups: rw.layer_groups || [],
  moeMode: rw.moe_mode || reg.recommended_moe_mode || 'masked_on_chip' },
  runtimeKnobs(gapDef)); }
  function currentWorkloadA() {
  const base = regimeToWorkload(STATE.pinA);
  if (!STATE.customDirty && REGIMES[STATE.pinA]) return base;
  const p = STATE.presets[parseInt(el('ctl-layout').value, 10) || 0] || base;
  const phase = el('ctl-phase').value, batch = parseInt(el('ctl-batch').value, 10) || 1;
  const isCh = phase === 'prefill' || phase === 'mixed';
  const rSeq = parseInt(el('ctl-seq-tokens').value, 10) || 1;
  const seqTokens = phase === 'decode' ? 1 : rSeq;
  const ctxVal = parseInt(el('ctl-context').value, 10) || 4096;
  const same = phase === base.phase && seqTokens === base.seqTokens &&
  ctxVal === base.context, actT = phase === 'mixed' ? batch + seqTokens
  : batch * seqTokens, lbl = `Live (${phase} B=${batch}, seq=${seqTokens})`;
  return Object.assign({}, base, { label: lbl, phase, batch, seqTokens, context: ctxVal,
  chunkSize: same ? base.chunkSize : (isCh ? seqTokens : 0),
  isl: same ? base.isl : (isCh ? Math.max(seqTokens, ctxVal) : ctxVal),
  specTokens: phase === 'verify' ? Math.max(1, seqTokens - 1) : 0,
  prefixHitRate: phase === base.phase ? base.prefixHitRate : 0.0,
  tp: p.tp, ep: p.ep, dp: p.dp, pp: p.pp, dcp: p.dcp, pcp: p.pcp,
  layerGroups: (p.layerGroups && p.layerGroups.length) ? p.layerGroups
  : base.layerGroups, moeMode: (same && batch === base.batch) ? base.moeMode
  : (actT >= 32 ? 'grouped_on_chip' : 'masked_on_chip') },
  runtimeKnobs(base.launchGapUs)); }
  function syncControlsFromRegime(idx) { const wl = regimeToWorkload(idx);
  el('ctl-phase').value = wl.phase; el('ctl-batch').value = Math.min(1024, wl.batch);
  el('ctl-seq-tokens').value = Math.min(8192, wl.seqTokens);
  el('ctl-context').value = Math.min(131072, wl.context);
  const pIdx = STATE.presets.findIndex(p => p.tp === wl.tp && p.ep === wl.ep &&
  p.dp === wl.dp && p.pp === wl.pp && p.dcp === wl.dcp && p.pcp === wl.pcp);
  if (pIdx >= 0 && el('ctl-layout')) el('ctl-layout').value = String(pIdx);
  if (!STATE.calibrated && !STATE.hwDirty)
  el('ctl-launch-gap').value = String(wl.launchGapUs); updateControlLabels(wl); }
  function updateControlLabels(wl) {
  el('lbl-phase').textContent = wl.phase; el('lbl-batch').textContent = wl.batch;
  el('lbl-seq-tokens').textContent = wl.seqTokens;
  el('lbl-context').textContent = (wl.context / 1024).toFixed(1) + 'K';
  el('lbl-vmem').textContent = wl.vmemMip.toFixed(1) + ' MiB';
  el('lbl-dma').textContent = wl.dmaBuffers + ' slots';
  el('lbl-alias').textContent = wl.vmemAlias ? 'enabled' : 'disabled';
  el('lbl-launch-gap').textContent = wl.launchGapUs.toFixed(2) + ' us';
  el('lbl-coll-hop').textContent = wl.collHopUs.toFixed(2) + ' us';
  const axes = [`TP${wl.tp}`]; for (const k of ['dp', 'pp', 'dcp', 'pcp', 'ep'])
  if (wl[k] > 1) axes.push(`${k.toUpperCase()}${wl[k]}`);
  const layStr = axes.join(' x ');
  el('lbl-layout').textContent = layStr; el('wb-active-layout').textContent = layStr;
  } function renderVmemArena(evalA) {
  const scopeSel = el('ctl-arena-scope').value || evalA.winner.scope;
  const sc = evalA.scopes.find(s => s.scope === scopeSel) || evalA.winner;
  const budget = Math.max(1.0, evalA.vmemBudgetMib), W = 760, padL = 16, barW = 728;
  const used = sc.dense_stage_mib + sc.expert_stage_mib + sc.op_scratch_mib +
  sc.pad_mib + sc.spine_dyn_mib + sc.op_dyn_mib, selT = STATE.selectedTopic || {};
  const px = m => (m / Math.max(budget * 1.12, used * 1.05)) * barW;
  const blocks = [['dense','Dense Ring',sc.dense_stage_mib,'#3B82F6'],
  ['exp','Expert Ring',sc.expert_stage_mib,'#6366F1'],
  ['scr','Op Scratch',sc.op_scratch_mib,'#0EA5E9'],
  ['pad','Align Reserve',sc.pad_mib,'#64748B'],
  ['spn','[T,H] Spine',sc.spine_dyn_mib,'#10B981'],
  ['tok','Token Scratch',sc.op_dyn_mib,'#F59E0B']]; let curX = padL, rects = '';
  for (const [key, label, mib, col] of blocks) { if (mib <= 0.01) continue;
  const w = Math.max(2, px(mib)), mid = (curX + w / 2).toFixed(1);
  const stk = (selT.kind === 'vmem' && selT.sub === key)
  ? ' stroke="#FFF" stroke-width="2"' : '';
  const lbl = w > 58 ? `<text x="${mid}" y="45" text-anchor="middle" fill="#FFF"` +
  ` font-size="11" font-weight="600">${label}</text>` : '';
  rects += `<g class="wb-clickable" onclick="wbSelectTopic('vmem','${sc.scope}',` +
  `'${key}')">${rSvg(curX, 22, w, 36, col).replace('/>', ` rx="3"${stk}><title>` +
  `${label}: ${mib.toFixed(2)} MiB</title></rect>`)}${lbl}</g>`; curX += w; }
  const ceilX = (padL + px(budget)).toFixed(1), ceilLblX = (+ceilX - 6).toFixed(1);
  const stCol = sc.n_splits > 1 ? '#F59E0B' : '#10B981', stTxt = sc.n_splits > 1
  ? `Multi-pass split: ${sc.n_splits} passes (T_max=${sc.t_vmem_max})`
  : `Single-pass resident: ${evalA.activeTokens}/${sc.t_vmem_max} tok ` +
  `(${used.toFixed(1)}/${budget.toFixed(1)} MiB)`;
  el('wb-vmem-arena-svg').innerHTML = `<svg viewBox="0 0 ${W} 96" class="chart-svg"` +
  ` role="img" aria-label="VMEM Floorplan"><rect x="${padL}" y="22" width="${barW}"` +
  ` height="36" rx="4" fill="var(--chart-bg)" stroke="var(--border)"/>${rects}` +
  `<line x1="${ceilX}" y1="12" x2="${ceilX}" y2="68" stroke="#EF4444" ` +
  `stroke-width="2" stroke-dasharray="4,3"/><text x="${ceilLblX}" y="14" ` +
  `text-anchor="end" fill="#EF4444" font-size="11" font-weight="700">Ceiling ` +
  `${budget.toFixed(1)} MiB</text><text x="${padL}" y="82" fill="${stCol}" ` +
  `font-size="12" font-weight="600">${stTxt} &middot; Aliasing saves ` +
  `${sc.alias_saved_kib} KiB/tok</text></svg>`; }
  function renderScopeTimeline(evalA) { const W = 760, padL = 112, barArea = 526;
  const feasHi = evalA.scopes.map(s => s.feasible ? s.step_upper_us : 0);
  const maxUs = Math.max(1.0, ...feasHi) * 1.06, px = u => (u / maxUs) * barArea;
  let rows = ''; ['whole_step', 'layer_group', 'single_layer', 'per_op']
  .forEach((nm, idx) => { const s = evalA.scopes.find(x => x.scope === nm);
  if (!s) return; const y = 18 + idx * 48, isWin = s.scope === evalA.winner.scope
  && s.feasible; if (!s.feasible) { rows += `<text x="10" y="${y + 18}" ` +
  `fill="var(--muted)" font-size="12">${s.scope}</text><text x="${padL}" ` +
  `y="${y + 18}" fill="var(--muted)" font-size="11">Infeasible: ` +
  `${escH(s.reason)}</text>`; return; }
  const envW = px(Math.max(s.compute_us, s.hbm_us)), commW = px(s.comm_us);
  const fdW = px(s.fill_drain_us), repW = px(s.repass_spill_us);
  const x1 = padL + envW, x2 = x1 + commW, x3 = x2 + fdW, x4 = x3 + repW;
  const hiX = padL + px(s.step_upper_us), loX = padL + px(s.step_bound_us);
  const bars = rSvg(padL, y + 2, Math.max(1, px(s.compute_us)), 12, '#3B82F6') +
  rSvg(padL, y + 16, Math.max(1, px(s.hbm_us)), 12, '#009E73') +
  rSvg(x1, y + 2, commW, 26, '#E69F00') + rSvg(x2, y + 2, fdW, 26, '#56B4E9') +
  rSvg(x3, y + 2, repW, 26, '#EF4444') +
  rSvg(x4, y + 2, px(s.launch_us), 26, '#CC79A7');
  rows += `<g class="wb-clickable" onclick="wbSelectTopic('scope','${s.scope}',` +
  `'total')"><text x="10" y="${y + 15}" fill="${isWin ? '#10B981' : 'var(--text)'}"` +
  ` font-size="12" font-weight="${isWin ? 700 : 500}">${isWin ? '* ' : ''}` +
  `${s.scope}</text><text x="10" y="${y + 29}" fill="var(--muted)" font-size="10">` +
  `${s.ops_per_step} ops, ${s.n_splits}p</text>${bars}<line x1="${loX.toFixed(1)}"` +
  ` y1="${y + 15}" x2="${hiX.toFixed(1)}" y2="${y + 15}" stroke="var(--text)" ` +
  `stroke-width="1.5" stroke-dasharray="2,2"/><text x="${(hiX + 6).toFixed(1)}" ` +
  `y="${y + 19}" fill="var(--text)" font-size="11" font-weight="600" class="tabular">` +
  `${s.step_bound_us.toFixed(1)} ~ ${s.step_upper_us.toFixed(1)} us</text></g>`; });
  const leg = [['#3B82F6',0,'Compute'],['#009E73',78,'HBM'],
  ['#E69F00',130,'Collectives'],['#56B4E9',214,'Fill/Drain'],
  ['#EF4444',292,'Repass'],['#CC79A7',356,'Launch Gap']].map(([c, x, t]) =>
  `<rect x="${x}" y="-8" width="8" height="8" fill="${c}"/>` +
  `<text x="${x + 11}" y="0">${t}</text>`).join('');
  el('wb-scope-timeline-svg').innerHTML = `<svg viewBox="0 0 ${W} 224" ` +
  `class="chart-svg" role="img" aria-label="4-Scope Step Timeline">${rows}` +
  `<g transform="translate(${padL}, 214)" font-size="10" ` +
  `fill="var(--muted)">${leg}</g></svg>`; }
  function renderSignedWaterfall(evalA, evalB, labelB) {
  const scA = evalA.winner, perOp = evalA.scopes.find(s => s.scope === 'per_op');
  const scB = evalB ? evalB.winner : (perOp || evalA.runnerUp);
  const cmpB = evalB ? `${labelB} (${scB.scope})` : `Baseline (${scB.scope})`;
  el('wb-waterfall-title').textContent =
  `Signed Attribution: [A] ${scA.scope} vs [B] ${cmpB}`;
  const envD = Math.max(scA.compute_us, scA.hbm_us) -
  Math.max(scB.compute_us, scB.hbm_us), selT = STATE.selectedTopic || {};
  const terms = [['Max(Compute, HBM) Envelope', envD],
  ['Collective Sync & Bounce', scA.comm_us - scB.comm_us],
  ['Cold Pipeline Fill/Drain', scA.fill_drain_us - scB.fill_drain_us],
  ['Multi-Pass HBM Repass Spill', scA.repass_spill_us - scB.repass_spill_us],
  ['Host Kernel Launch Gaps', scA.launch_us - scB.launch_us]];
  const netD = scA.step_bound_us - scB.step_bound_us;
  const pct = scB.step_bound_us > 0 ? (netD / scB.step_bound_us) * 100.0 : 0.0;
  const sgn = v => (v >= 0 ? '+' : '') + v.toFixed(1);
  el('wb-waterfall-summary').textContent = `Net: ${sgn(netD)} us (${sgn(pct)}%)`;
  const maxAbs = Math.max(1.0, ...terms.map(t => Math.abs(t[1])), Math.abs(netD));
  const W = 760, midX = 430, halfW = 220; const rows = terms.map(([lbl, d], i) => {
  const y = 16 + i * 28, w = Math.max(1.5, (Math.abs(d) / maxAbs) * halfW);
  const x = d < 0 ? midX - w : midX, col = d <= 0 ? '#10B981' : '#EF4444';
  const stk = (selT.kind === 'waterfall' && String(selT.sub) === String(i))
  ? ' stroke="#FFF" stroke-width="2"' : '', bar = rSvg(x, y + 2, w, 16, col)
  .replace('/>', stk + '/>');
  return `<g class="wb-clickable" onclick="wbSelectTopic('waterfall',` +
  `'${scA.scope}','${i}')"><text x="12" y="${y + 14}" fill="var(--text)" ` +
  `font-size="11">${lbl}</text>${bar}<text x="${W - 14}" y="${y + 14}" ` +
  `text-anchor="end" fill="${col}" font-size="11" font-weight="600" ` +
  `class="tabular">${d > 0 ? '+' : ''}${d.toFixed(2)} us</text></g>`; }).join('');
  el('wb-waterfall-svg').innerHTML = `<svg viewBox="0 0 ${W} 162" ` +
  `class="chart-svg" role="img" aria-label="Signed Mechanism Waterfall">` +
  `<line x1="${midX}" y1="8" x2="${midX}" y2="154" stroke="var(--border)" ` +
  `stroke-width="1.5"/>${rows}</svg>`; }
  function renderDerivDrawer(wlA, evalA, evalB, lblB) {
  const selSc = (STATE.selectedTopic || {}).scope;
  const sc = evalA.scopes.find(s => s.scope === selSc) || evalA.winner;
  const provTag = STATE.calibrated ? 'MEASURED' : 'ASSUMED';
  const bind = sc.compute_us >= sc.hbm_us ? 'Matrix Compute' : 'HBM Stream';
  const cal = window.wbCalNotes ? window.wbCalNotes(sc) : { line:
  '<span class="prov-tag prov-tag-ASSUMED">ASSUMED</span> Settle: --trace &lt;p&gt;',
  anc: '' }, spK = (evalA.vmemParts.spineBytes / 1024).toFixed(1);
  const opK = (evalA.vmemParts.opBytes / 1024).toFixed(1);
  const totK = (evalA.vmemParts.totalPerTokenBytes / 1024).toFixed(1);
  const focus = window.wbFocusedTopicCard ? window.wbFocusedTopicCard(wlA, evalA,
  evalB, lblB, STATE.selectedTopic, sc) : '';
  const b1 = `<div class="kpi-sub" style="margin-top:4px;">Binding: <strong>${bind}` +
  `</strong> &middot; Pad: <strong class="tabular">${sc.moe_flop_factor.toFixed(3)}x` +
  `</strong></div><div class="wb-formula">t_step = max(t_comp, t_hbm) + t_comm + ` +
  `t_fd + t_rep + N*t_gap
= max(${sc.compute_us.toFixed(2)}, ${sc.hbm_us.toFixed(2)})` +
  ` + ${sc.comm_us.toFixed(2)} + ${sc.fill_drain_us.toFixed(2)} + ` +
  `${sc.repass_spill_us.toFixed(2)} + ${sc.ops_per_step}*` +
  `${wlA.launchGapUs.toFixed(2)}
= ${sc.step_bound_us.toFixed(2)} us ` +
  `(upper: ${sc.step_upper_us.toFixed(2)} us)</div>`;
  const b2 = `<div class="wb-formula">T_max = floor((B_vmem - B_static) / b_tok)
` +
  `B_static = ${sc.dense_stage_mib}+${sc.expert_stage_mib}+${sc.op_scratch_mib}+` +
  `${sc.pad_mib} = ${sc.vmem_static_mib.toFixed(2)} MiB
b_tok = ${spK}+${opK} = ` +
  `${totK} KiB/tok -> T_max=${sc.t_vmem_max} (${sc.n_splits}p)</div>`;
  const b3 = `<div class="wb-formula">launch_gap = ${wlA.launchGapUs.toFixed(2)} ~ ` +
  `${STATE.launchGapHiUs.toFixed(2)} us
coll_hop = ${wlA.collHopUs.toFixed(2)} ~ ` +
  `${STATE.collHopHiUs.toFixed(2)} us
w_step = ${sc.w_step_gb} GB | act = ` +
  `${sc.act_hbm_mb} MB</div><div style="font-size:11px;margin-top:4px;">` +
  `${cal.line}</div>`;
  el('wb-deriv-drawer').innerHTML = focus + dBox(`Scope: ${sc.scope}`, 'DERIVED', b1) +
  dBox('VMEM Ceiling', 'DERIVED', b2) +
  dBox(`Priors (${sc.ops_per_step} ops)`, provTag, b3) + cal.anc; }
  function renderRegimeList() {
  const listEl = el('wb-regime-list'); if (!listEl) return;
  const vsBtn = el('btn-vs-perop');
  if (vsBtn) vsBtn.className = 'wb-mini-btn' + (STATE.pinB < 0 ? ' active-b' : '');
  listEl.innerHTML = REGIMES.map((r, i) => {
  const ev = evaluateWorkload(regimeToWorkload(i));
  const isA = i === STATE.pinA && !STATE.customDirty, isB = i === STATE.pinB;
  const cls = (isA ? ' pin-a' : '') + (isB ? ' pin-b' : '');
  const col = SCOPE_COLORS[ev.winner.scope] || '#94A3B8';
  return `<div class="wb-regime-item${cls}" onclick="wbPinRegime(${i}, 'A')">` +
  `<div style="min-width:0;flex:1;"><div style="font-weight:600;white-space:nowrap;` +
  `overflow:hidden;text-overflow:ellipsis;">${escH(r.label)}</div><div style=` +
  `"color:var(--muted);font-size:10px;"><span style="color:${col};font-weight:700;">` +
  `${escH(ev.winner.scope)}</span> &middot; T=${ev.activeTokens} &middot; ` +
  `${ev.winner.step_bound_us} us</div></div><button class="wb-mini-btn` +
  `${isA ? ' active-a' : ''}" type="button" onclick="event.stopPropagation();` +
  `wbPinRegime(${i}, 'A')">A</button><button class="wb-mini-btn` +
  `${isB ? ' active-b' : ''}" type="button" onclick="event.stopPropagation();` +
  `wbPinRegime(${i}, 'B')">B</button></div>`; }).join(''); }
  window.wbPinRegime = function(idx, slot) { if (slot === 'B') {
  if (STATE.pinB === idx) STATE.pinB = -1; else { STATE.pinB = idx;
  if (STATE.pinA === idx && REGIMES.length > 1) { STATE.pinA = idx === 0 ? 1 : 0;
  STATE.customDirty = false; syncControlsFromRegime(STATE.pinA); } } } else {
  STATE.pinA = idx; if (STATE.pinB === idx) STATE.pinB = -1; STATE.customDirty = false;
  syncControlsFromRegime(idx);
  const winSc = evaluateWorkload(regimeToWorkload(idx)).winner.scope;
  el('ctl-arena-scope').value = winSc;
  STATE.selectedTopic = { kind: 'scope', scope: winSc, sub: 'total' }; }
  window.wbRenderAll(); };
  window.wbSwapAB = function() {
  if (STATE.pinB < 0 || STATE.pinB === STATE.pinA) { STATE.pinB = STATE.pinA;
  STATE.pinA = (STATE.pinA + 1) % Math.max(1, REGIMES.length); } else {
  [STATE.pinA, STATE.pinB] = [STATE.pinB, STATE.pinA]; }
  STATE.customDirty = false; syncControlsFromRegime(STATE.pinA); window.wbRenderAll();
  }; window.wbClearB = function() { if (STATE.pinB < 0) {
  const arEl = el('ctl-arena-scope'), cur = arEl.value || 'whole_step';
  const nxt = cur === 'per_op' ? evaluateWorkload(currentWorkloadA()).winner.scope
  : 'per_op'; arEl.value = nxt;
  STATE.selectedTopic = { kind: 'scope', scope: nxt, sub: 'total' };
  } else STATE.pinB = -1; window.wbRenderAll(); };
  window.wbResetSliders = function() {
  STATE.pinA = 0; STATE.pinB = -1; STATE.resetSeq = (STATE.resetSeq || 0) + 1;
  STATE.customDirty = false; STATE.hwDirty = false; STATE.calibrated = false;
  el('ctl-vmem').value = String(Math.min(STATE.maxVmemMib,
  Math.max(2.0, DATA.vmem_budget_mib || STATE.maxVmemMib)));
  el('ctl-dma').value = String(DATA.dma_buffers || 2); el('ctl-alias').value = '1';
  el('ctl-coll-hop').value = ((CHIP.collective_latency_s || 1e-6) * 1e6).toFixed(2);
  syncControlsFromRegime(0);
  const winSc = evaluateWorkload(regimeToWorkload(0)).winner.scope;
  el('ctl-arena-scope').value = winSc;
  STATE.selectedTopic = { kind: 'scope', scope: winSc, sub: 'total' };
  window.wbRenderAll(); };
  window.wbOnPhaseChange = function() {
  if (el('ctl-phase').value === 'decode') el('ctl-seq-tokens').value = '1';
  STATE.customDirty = true; window.wbRenderAll(); };
  window.wbOnSeqChange = function() {
  const v = parseInt(el('ctl-seq-tokens').value, 10) || 1, phEl = el('ctl-phase');
  if (phEl.value === 'decode' && v > 1) phEl.value = v <= 16 ? 'verify' : 'prefill';
  STATE.customDirty = true; window.wbRenderAll(); };
  window.wbRenderAll = function() {
  const wlA = currentWorkloadA(); updateControlLabels(wlA);
  const evalA = evaluateWorkload(wlA), hasB = STATE.pinB >= 0 && REGIMES[STATE.pinB];
  const evalB = hasB ? evaluateWorkload(regimeToWorkload(STATE.pinB)) : null;
  const lblB = hasB ? regimeToWorkload(STATE.pinB).label : '';
  if (el('wb-ab-status')) el('wb-ab-status').innerHTML = `<strong>[A]</strong> ` +
  `${escH(wlA.label)} (${evalA.winner.scope}) vs <strong>[B]</strong> ` +
  (hasB ? `${escH(lblB)} (${evalB.winner.scope})` : 'per_op') +
  (STATE.resetSeq ? ` &middot; #${STATE.resetSeq}` : '');
  el('wb-kpi-scope').textContent = `${evalA.winner.scope} (${evalA.moeModeRec})`;
  el('wb-kpi-floor').textContent = `${evalA.winner.step_bound_us.toFixed(1)} ~ ` +
  `${evalA.winner.step_upper_us.toFixed(1)} us`; el('wb-kpi-tmax').textContent =
  `${evalA.winner.t_vmem_max} tok (${evalA.winner.n_splits}p)`;
  el('wb-kpi-tps').textContent = `${evalA.tpsChip.toFixed(1)} tok/s/chip`;
  renderRegimeList(); renderVmemArena(evalA); renderScopeTimeline(evalA);
  renderSignedWaterfall(evalA, evalB, lblB);
  renderDerivDrawer(wlA, evalA, evalB, lblB);
  if (window.wbRenderPhaseExtension) window.wbRenderPhaseExtension(wlA, evalA); };
  window.wbOnControl = function() { STATE.customDirty = true; window.wbRenderAll(); };
  window.wbOnHwControl = function() { STATE.hwDirty = true; window.wbRenderAll(); };
  window.wbSelectTopic = function(kind, scope, sub) {
  STATE.selectedTopic = { kind, scope, sub }; el('ctl-arena-scope').value = scope;
  window.wbRenderAll(); }; window.wbSetView = function(mode) {
  const isWb = mode === 'workbench';
  document.body.classList.toggle('wb-mode-active', isWb);
  el('view-workbench').classList.toggle('view-hidden', !isWb);
  el('view-audit').classList.toggle('view-hidden', isWb);
  el('btn-view-workbench').classList.toggle('active', isWb);
  el('btn-view-audit').classList.toggle('active', !isWb); };
  window.wbSetStageTab = function(tab) {
  const isAr = tab === 'arena', arEl = el('wb-stage-arena');
  const phEl = el('wb-stage-phase');
  if (arEl) arEl.style.display = isAr ? 'flex' : 'none';
  if (phEl) phEl.style.display = isAr ? 'none' : 'flex';
  if (el('btn-stage-arena')) el('btn-stage-arena').classList.toggle('active', isAr);
  if (el('btn-stage-phase')) el('btn-stage-phase').classList.toggle('active', !isAr); };
  window.__WB_API = { STATE, DATA, SCOPE_COLORS, escH, evaluateWorkload,
  currentWorkloadA, regimeToWorkload, syncControlsFromRegime }; initPresets();
  if (REGIMES.length > 0) syncControlsFromRegime(0);
  document.body.classList.add('wb-mode-active');"""
    return render_workbench_engine_js() + ui_js + "\n  window.wbRenderAll();\n})();\n"
