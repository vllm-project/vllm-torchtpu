// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/* Thin presentation-only UI/SVG renderer backed by Python advisor APIs. */
(function () {
  const seedEl = document.getElementById('advisor-data');
  if (!seedEl) return;
  const DATA = JSON.parse(seedEl.textContent || '{}');
  const DEFAULT_EFF = DATA.default_efficiency || {};
  const DEFAULT_VMEM = Number(DATA.vmem_budget_mib);
  const COLORS = { whole_step: '#009E73', layer_group: '#56B4E9', single_layer: '#E69F00', per_op: '#CC79A7' };
  const SCOPE_SHORT_LABELS = { whole_step: 'Whole', layer_group: 'Group', single_layer: 'Layer', per_op: 'Per-Op' };
  const S = {
    mode: DATA.mode || 'static', pinA: 0, pinB: -1, activeEval: DATA.default_eval || (DATA.regime_evals || [])[0] || null,
    compareEval: null, phaseMap: DATA.phase_map || { cells: [], t_grid: [], s_grid: [] },
    bucketSchedule: DATA.bucket_schedule || {}, phaseMapMode: 'recommended',
    selectedScope: null, dispatchRegime: 'compiled_xla', calibratedSource: null, timer: null,
  };
  const el = (id) => document.getElementById(id);
  const esc = (v) => String(v == null ? '' : v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const rect = (x, y, w, h, f) => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${h.toFixed(1)}" fill="${f}"/>`;
  const fmtCtx = (c) => (c >= 1048576 ? `${(c / 1048576).toFixed(0)}M` : `${Math.round(c / 1024)}K`);
  const num = (id, d) => Number((el(id) && el(id).value) || d);
  const setTxt = (id, t) => { const n = el(id); if (n) n.textContent = t; };
  const scopeOf = (rep, nm) => (rep.scopes || []).find((s) => s.scope === nm) || (rep.scopes || [])[0];
  function setBanner(id, msg) {
    const b = el(id);
    if (!b) return;
    b.textContent = msg || '';
    b.classList.toggle('view-hidden', !msg);
  }
  function readPayload() {
    const phase = (el('ctl-phase') && el('ctl-phase').value) || 'decode';
    const seqTok = num('ctl-seq-tokens', 1), ctx = num('ctl-context', 4096);
    return {
      phase, batch: num('ctl-batch', 1), isl: phase === 'prefill' ? Math.max(seqTok, ctx) : ctx,
      context: ctx, chunk_size: phase === 'prefill' ? seqTok : 0, spec_tokens: phase === 'verify' ? Math.max(0, seqTok - 1) : 0,
      tp: num('ctl-tp', 8), ep: num('ctl-ep', 8), dp: num('ctl-dp', 1), pp: num('ctl-pp', 1),
      dcp: num('ctl-dcp', 1), pcp: num('ctl-pcp', 1), vmem_budget_mib: num('ctl-vmem', DEFAULT_VMEM),
      dma_buffers: num('ctl-dma', 2), vmem_alias: (el('ctl-alias') ? el('ctl-alias').value : '1') === '1',
      dispatch_regime: S.dispatchRegime, eta_dense: num('ctl-eta-dense', Number(DEFAULT_EFF.eta_dense)),
      eta_moe_grouped: num('ctl-eta-moe', Number(DEFAULT_EFF.eta_moe_grouped)),
      eta_hbm: num('ctl-eta-hbm', Number(DEFAULT_EFF.eta_hbm)), eta_ici: num('ctl-eta-ici', Number(DEFAULT_EFF.eta_ici)),
    };
  }
  function syncLabels(w) {
    setTxt('lbl-batch', String(w.batch));
    setTxt('lbl-seq-tokens', String(w.phase === 'prefill' ? (w.chunk_size || w.isl) : (1 + (w.spec_tokens || 0))));
    setTxt('lbl-context', fmtCtx(w.context || 4096));
    setTxt('lbl-vmem', `${Number(w.vmem_budget_mib).toFixed(1)} MiB`);
    setTxt('lbl-dispatch-regime', S.dispatchRegime);
    for (const [id, v] of [['eta-dense', w.eta_dense], ['eta-moe', w.eta_moe_grouped], ['eta-hbm', w.eta_hbm], ['eta-ici', w.eta_ici]]) {
      setTxt(`lbl-${id}`, Number(v).toFixed(2));
    }
  }
  function syncFromRegime(idx) {
    const reg = (DATA.regimes || [])[idx];
    if (!reg || !reg.workload) return;
    const w = reg.workload;
    const sv = (id, v) => { const n = el(id); if (n && v != null) n.value = String(v); };
    sv('ctl-phase', w.phase || 'decode'); sv('ctl-batch', w.batch);
    sv('ctl-context', reg.effective_context || w.context || w.isl || 4096);
    sv('ctl-seq-tokens', w.phase === 'prefill' ? (w.chunk_size || w.isl || 512) : (1 + (w.spec_tokens || 0)));
    for (const ax of ['tp', 'ep', 'dp', 'pp', 'dcp', 'pcp']) sv(`ctl-${ax}`, w[ax] || 1);
    syncLabels(readPayload());
  }
  function renderTopBar(ev) {
    const rep = ev.report, pill = el('wb-verdict-pill'), sumEl = el('wb-verdict-summary');
    if (!rep || !pill || !sumEl) return;
    const rec = scopeOf(rep, rep.recommended_scope), po = scopeOf(rep, 'per_op');
    const invalid = String(rep.verdict || '').startsWith('invalid_config(');
    pill.textContent = invalid ? 'INVALID CONFIG' : rep.recommended_scope.toUpperCase();
    pill.style.background = invalid ? '#DC2626' : (COLORS[rep.recommended_scope] || '#475569');
    const save = po && rec ? (po.step_achievable_us - rec.step_achievable_us) : 0.0;
    const pct = po && po.step_achievable_us > 1e-6 ? (save / po.step_achievable_us) * 100.0 : 0.0;
    const coreSupport = rep.core_support || (rec && rec.support) || 'baseline';
    const coreReadyScope = rep.core_ready_scope || 'per_op';
    const coreBlockers = Array.isArray(rep.core_blockers) ? rep.core_blockers : [];
    const coreBlkHtml = (coreSupport === 'outside_core_limits' || coreBlockers.length)
      ? ` &middot; Blocked: <code>${esc(coreBlockers.join(', ') || 'none')}</code>`
      : '';
    sumEl.innerHTML = `Saves <strong>${save.toFixed(1)} us (${pct.toFixed(1)}% achievable)</strong> &middot; Fastest: <strong>${esc(rep.fastest_scope)}</strong> &middot; Core: <code>${esc(coreSupport)}</code> (ready: <code>${esc(coreReadyScope)}</code>)${coreBlkHtml} &middot; Verdict: <code>${esc(rep.verdict)}</code>`;
    setTxt('wb-kpi-scope', rep.recommended_scope);
    setTxt('wb-kpi-core', coreSupport);
    setTxt('wb-kpi-floor', rec ? `${Number(rec.step_bound_us).toFixed(1)} us` : '-');
    setTxt('wb-kpi-tmax', rec ? `${rec.max_vmem_tokens} tok` : '-');
    const tps = (rep.metrics || {}).throughput_tok_s_per_chip;
    setTxt('wb-kpi-tps', tps ? `${Number(tps.value).toFixed(1)}` : '-');
    setTxt('wb-kpi-bind', rep.binding_unit || (rec && rec.binding_unit) || '-');
    setBanner('wb-invalid-banner', invalid ? `FEASIBILITY RED FLAG: ${rep.verdict}` : (!rep.hbm_fits ? 'FEASIBILITY RED FLAG: PHYSICAL HBM OOM' : ''));
  }
  function renderArena(ev) {
    const rep = ev.report, target = el('wb-vmem-arena-svg');
    if (!rep || !target) return;
    const sc = scopeOf(rep, (el('ctl-arena-scope') && el('ctl-arena-scope').value) || rep.recommended_scope);
    if (!sc) return;
    const bud = Math.max(1.0, Number(rep.vmem_budget_mib || DEFAULT_VMEM)), st = Number(sc.vmem_static_mib || 0.0);
    const dyn = (Number(rep.per_token_vmem_bytes || 0.0) * Number(rep.active_tokens || 1)) / 1048576.0;
    const used = st + dyn, scale = Math.max(bud * 1.12, used * 1.05), px = (m) => (m / scale) * 728;
    const stW = px(st), dynW = px(dyn), ceilX = 16 + px(bud);
    const col = sc.num_vmem_splits > 1 ? '#F59E0B' : '#10B981';
    const txt = `${rep.active_tokens}/${sc.max_vmem_tokens} tok (${used.toFixed(1)}/${bud.toFixed(1)} MiB, ${sc.num_vmem_splits} passes)`;
    target.innerHTML = `<svg viewBox="0 0 760 96" class="chart-svg" role="img" aria-label="VMEM Floorplan">`
      + `<rect x="16" y="22" width="728" height="36" rx="4" fill="var(--chart-bg)" stroke="var(--border)"/>`
      + rect(16, 22, stW, 36, '#3B82F6') + rect(16 + stW, 22, dynW, 36, '#10B981')
      + `<line x1="${ceilX.toFixed(1)}" y1="12" x2="${ceilX.toFixed(1)}" y2="68" stroke="#EF4444" stroke-width="2" stroke-dasharray="4,3"/>`
      + `<text x="${(ceilX - 6).toFixed(1)}" y="14" text-anchor="end" fill="#EF4444" font-size="11" font-weight="700">Budget ${bud.toFixed(1)} MiB</text>`
      + `<text x="16" y="82" fill="${col}" font-size="12" font-weight="600">${esc(txt)}</text></svg>`;
  }
  function renderTimeline(ev) {
    const rep = ev.report, target = el('wb-scope-timeline-svg');
    if (!rep || !target) return;
    const scopes = rep.scopes || [];
    const maxUs = Math.max(1.0, ...scopes.map((s) => (s.feasible ? Math.max(s.step_upper_us, s.step_achievable_us) : 0))) * 1.08;
    const px = (u) => (Number(u || 0) / maxUs) * 480;
    const rows = ['whole_step', 'layer_group', 'single_layer', 'per_op'].map((nm, idx) => {
      const s = scopes.find((x) => x.scope === nm);
      if (!s) return '';
      const y = 18 + idx * 46, win = s.scope === rep.recommended_scope && s.feasible;
      if (!s.feasible) return `<text x="10" y="${y + 18}" fill="var(--muted)" font-size="12">${esc(s.scope)}</text><text x="112" y="${y + 18}" fill="var(--muted)" font-size="11">Infeasible: ${esc(s.reason)}</text>`;
      const loX = 112 + px(s.step_bound_us), hiX = 112 + px(s.step_upper_us), achX = 112 + px(s.step_achievable_us);
      return `<g class="wb-clickable" onclick="wbSelectScope('${s.scope}')">`
        + `<text x="10" y="${y + 15}" fill="${win ? '#10B981' : 'var(--text)'}" font-size="12" font-weight="${win ? 700 : 500}">${win ? '* ' : ''}${esc(s.scope)}</text>`
        + `<text x="10" y="${y + 29}" fill="var(--muted)" font-size="10">${s.ops_per_step} ops, ${s.num_vmem_splits} passes</text>`
        + rect(112, y + 4, Math.max(2, px(s.compute_us)), 10, '#3B82F6') + rect(112, y + 16, Math.max(2, px(s.hbm_us)), 10, '#009E73')
        + `<line x1="${loX.toFixed(1)}" y1="${y + 14}" x2="${hiX.toFixed(1)}" y2="${y + 14}" stroke="var(--text)" stroke-width="2"/>`
        + `<circle cx="${achX.toFixed(1)}" cy="${y + 14}" r="4" fill="#F59E0B"/>`
        + `<text x="${(Math.max(hiX, achX) + 8).toFixed(1)}" y="${y + 14}" fill="var(--text)" font-size="10" font-weight="600">${s.step_bound_us.toFixed(1)}~${s.step_upper_us.toFixed(1)} us</text>`
        + `<text x="${(Math.max(hiX, achX) + 8).toFixed(1)}" y="${y + 27}" fill="var(--muted)" font-size="9">achievable ${s.step_achievable_us.toFixed(1)} us</text></g>`;
    }).join('');
    target.innerHTML = `<svg viewBox="0 0 760 214" class="chart-svg" role="img" aria-label="4-Scope Step Timeline">${rows}</svg>`;
  }
  function renderWaterfall(evA, evB) {
    const repA = evA.report, target = el('wb-waterfall-svg');
    if (!repA || !target) return;
    const scA = scopeOf(repA, repA.fastest_scope);
    const scB = evB ? scopeOf(evB.report, evB.report.fastest_scope) : scopeOf(repA, 'per_op');
    if (!scA || !scB) return;
    const baseTerms = [
      ['Compute Chain Delta', scA.compute_us - scB.compute_us],
      ['HBM Stream Delta', scA.hbm_us - scB.hbm_us],
      ['Collective Comm Delta', scA.comm_us - scB.comm_us],
      ['Launch, Host & Fill/Drain Delta', (scA.launch_us + scA.fill_drain_us + (scA.host_head_us || 0)) - (scB.launch_us + scB.fill_drain_us + (scB.host_head_us || 0))],
    ];
    const netD = scA.step_bound_us - scB.step_bound_us;
    const rawSum = baseTerms.reduce((acc, item) => acc + item[1], 0.0);
    const terms = [...baseTerms, ['Overlap / Parallelism Delta', netD - rawSum]];
    setTxt('wb-waterfall-title', `Signed Attribution: ${scA.scope} vs ${scB.scope}`);
    setTxt('wb-waterfall-summary', `Net floor delta: ${netD >= 0 ? '+' : ''}${netD.toFixed(1)} us`);
    const maxAbs = Math.max(1.0, ...terms.map((t) => Math.abs(t[1])), Math.abs(netD));
    const rows = terms.map(([lbl, val], idx) => {
      const y = 16 + idx * 28, w = Math.max(1.5, (Math.abs(val) / maxAbs) * 205), x = val < 0 ? 430 - w : 430;
      const col = val <= 0 ? '#10B981' : '#EF4444';
      return `<text x="12" y="${y + 14}" fill="var(--text)" font-size="11">${esc(lbl)}</text>${rect(x, y + 2, w, 16, col)}`
        + `<text x="746" y="${y + 14}" text-anchor="end" fill="${col}" font-size="11" font-weight="600">${val >= 0 ? '+' : ''}${val.toFixed(2)} us</text>`;
    }).join('');
    target.innerHTML = `<svg viewBox="0 0 760 162" class="chart-svg" role="img" aria-label="Signed Attribution Waterfall"><line x1="430" y1="8" x2="430" y2="154" stroke="var(--border)" stroke-width="1.5"/>${rows}</svg>`;
  }
  function renderInspector(ev) {
    const rep = ev.report, target = el('wb-deriv-drawer');
    if (!rep || !target) return;
    const sc = scopeOf(rep, S.selectedScope || rep.recommended_scope);
    if (!sc) return;
    const blk = (sc.blockers || []).length ? ` (${esc(sc.blockers.join(', '))})` : '';
    const coreSupport = rep.core_support || sc.support || 'baseline';
    const coreReadyScope = rep.core_ready_scope || 'per_op';
    const coreBlockers = Array.isArray(rep.core_blockers) ? rep.core_blockers : [];
    const coreBlk = (coreSupport === 'outside_core_limits' || coreBlockers.length)
      ? ` (${esc(coreBlockers.join(', ') || 'none')})`
      : '';
    const cal = S.calibratedSource ? `<span class="prov-tag prov-tag-MEASURED">MEASURED</span> ${esc(S.calibratedSource)}` : '<span class="prov-tag prov-tag-ASSUMED">ASSUMED</span> Hardware priors';
    target.innerHTML = `<div class="wb-exec-card"><div class="wb-exec-row"><span>Inspecting</span><strong>${esc(sc.scope)} (${sc.step_achievable_us.toFixed(1)} us)</strong></div>`
      + `<div class="wb-exec-row"><span>Scope Support</span><strong>${esc(sc.support)}${blk}</strong></div>`
      + `<div class="wb-exec-row"><span>Core Readiness</span><strong>${esc(coreSupport)} (${esc(coreReadyScope)})${coreBlk}</strong></div>`
      + `<div class="wb-exec-row"><span>Binding</span><strong>${esc(sc.binding_unit || '-')}</strong></div>`
      + `<div class="wb-exec-row"><span>Calibration</span><span>${cal}</span></div></div>`
      + `<details class="wb-acc" open><summary>Step-Time Breakdown (Python Server)</summary><div class="wb-acc-body">`
      + `<div class="wb-formula">strict=${sc.step_floor_strict_us.toFixed(2)} | floor=${sc.step_bound_us.toFixed(2)} | upper=${sc.step_upper_us.toFixed(2)} us
`
      + `achievable=${sc.step_achievable_us.toFixed(2)}~${sc.step_achievable_upper_us.toFixed(2)} us
`
      + `compute=${sc.compute_us.toFixed(2)} | hbm=${sc.hbm_us.toFixed(2)} | comm=${sc.comm_us.toFixed(2)} | launch=${sc.launch_us.toFixed(2)} | host=${Number(sc.host_head_us || 0).toFixed(2)} us
`
      + `max_vmem_tokens=${sc.max_vmem_tokens} | splits=${sc.num_vmem_splits} | weight_step=${sc.step_weight_gb} GB</div>`
      + `<p style="font-size:11px;margin:4px 0;">${esc(rep.rationale)}</p></div></details>`;
  }
  function renderPhaseMap() {
    setTxt('wb-bucket-schedule-pre', JSON.stringify(S.bucketSchedule || {}, null, 2));
    const svgEl = el('wb-phase-map-svg');
    if (!svgEl) return;
    const pm = S.phaseMap || {}, tG = pm.t_grid || [], sG = pm.s_grid || [], cmap = new Map();
    for (const c of (pm.cells || [])) cmap.set(`${c.tokens}:${c.context}`, c);
    const W = 86 + Math.max(1, tG.length) * 84, H = 68 + Math.max(1, sG.length) * 30;
    const modeLabel = S.phaseMapMode === 'core_ready' ? 'Core-Ready Today' : 'Physics Winner';
    let body = `<text x="70" y="14" fill="var(--muted)" font-size="10">${esc(modeLabel)} | Legend: Whole = whole_step | Group = layer_group | Layer = single_layer | Per-Op = per_op | OOM = HBM OOM</text>`;
    body += tG.map((t, i) => `<text x="${70 + i * 84 + 42}" y="34" text-anchor="middle" fill="var(--muted)" font-size="11">T=${t}</text>`).join('');
    sG.forEach((s, si) => {
      const y = 46 + si * 30;
      body += `<text x="62" y="${y + 19}" text-anchor="end" fill="var(--muted)" font-size="11">${fmtCtx(s)}</text>`;
      tG.forEach((t, ti) => {
        const c = cmap.get(`${t}:${s}`), x = 70 + ti * 84;
        const scN = c ? (S.phaseMapMode === 'core_ready' ? c.core_ready_scope : c.recommended_scope) : 'per_op';
        const col = (c && !c.hbm_fits) ? '#7F1D1D' : (COLORS[scN] || '#475569');
        const txt = (c && !c.hbm_fits) ? 'OOM' : (SCOPE_SHORT_LABELS[scN] || scN);
        const tip = c
          ? `T=${t}, S=${fmtCtx(s)} | physics=${c.recommended_scope} | core_ready=${c.core_ready_scope || 'per_op'} | core_support=${c.core_support || 'baseline'}${(c.core_blockers && c.core_blockers.length) ? ` | blockers=${c.core_blockers.join(', ')}` : ''}`
          : `T=${t}, S=${fmtCtx(s)}`;
        body += `<g class="wb-clickable" onclick="wbSelectPhaseCell(${t},${s})"><title>${esc(tip)}</title><rect x="${x + 2}" y="${y + 2}" width="80" height="26" rx="4" fill="${col}" opacity="0.88"/>`
          + `<text x="${x + 42}" y="${y + 19}" text-anchor="middle" fill="#FFF" font-size="10" font-weight="700">${esc(txt)}</text></g>`;
      });
    });
    svgEl.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart-svg" role="img" aria-label="2D Phase Map">${body}</svg>`;
  }
  function renderAll() {
    const listEl = el('wb-regime-list');
    if (listEl) {
      listEl.innerHTML = (DATA.regimes || []).map((r, i) => {
        const isA = i === S.pinA, isB = i === S.pinB, col = COLORS[r.recommended_scope] || '#94A3B8';
        return `<div class="wb-regime-item${isA ? ' pin-a' : ''}${isB ? ' pin-b' : ''}" onclick="wbPinRegime(${i},'A')">`
          + `<div style="min-width:0;flex:1;"><div style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(r.label)}</div>`
          + `<div style="color:var(--muted);font-size:10px;"><span style="color:${col};font-weight:700;">${esc(r.recommended_scope)}</span> &middot; T=${r.active_tokens} &middot; ${Number(r.step_bound_us).toFixed(1)} us</div></div>`
          + `<button class="wb-mini-btn${isA ? ' active-a' : ''}" type="button" onclick="event.stopPropagation();wbPinRegime(${i},'A')">A</button>`
          + `<button class="wb-mini-btn${isB ? ' active-b' : ''}" type="button" onclick="event.stopPropagation();wbPinRegime(${i},'B')">B</button></div>`;
      }).join('');
    }
    if (S.activeEval) { renderTopBar(S.activeEval); renderArena(S.activeEval); renderTimeline(S.activeEval); renderWaterfall(S.activeEval, S.compareEval); renderInspector(S.activeEval); }
    renderPhaseMap();
  }
  async function postApi(route, payload) {
    const resp = await fetch(route, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    return { ok: resp.ok, data: await resp.json() };
  }
  async function requestAdvise() {
    const w = readPayload(); syncLabels(w);
    if (S.mode !== 'live') { setBanner('wb-static-banner', `Static snapshot: run "python -m vllm_torchtpu.megakernel.advisor serve --model ${DATA.model}" for live recomputation.`); return; }
    setBanner('wb-static-banner', '');
    const res = await postApi('/api/advise', w);
    if (res.ok) { S.activeEval = res.data; renderAll(); }
    else { setBanner('wb-invalid-banner', 'Error: ' + ((res.data && res.data.error) || 'request failed')); }
  }
  async function requestPhaseMap() {
    if (S.mode !== 'live') return;
    const res = await postApi('/api/phase_map', readPayload());
    if (res.ok) { S.phaseMap = res.data; if (res.data.bucket_schedule) S.bucketSchedule = res.data.bucket_schedule; renderPhaseMap(); }
    else { setBanner('wb-invalid-banner', 'Error: ' + ((res.data && res.data.error) || 'request failed')); }
  }
  window.wbOnControl = () => { clearTimeout(S.timer); S.timer = setTimeout(requestAdvise, 80); };
  window.wbSetDispatchRegime = (regime) => {
    S.dispatchRegime = regime;
    document.querySelectorAll('.wb-dr-group button').forEach((btn) => {
      btn.classList.toggle('active-a', btn.dataset.regime === regime);
    });
    requestAdvise();
  };
  window.wbSelectScope = (sc) => { S.selectedScope = sc; if (S.activeEval) renderInspector(S.activeEval); };
  window.wbPinRegime = (index, slot) => {
    const r = (DATA.regimes || [])[index]; if (!r) return;
    const evalPayload = (DATA.regime_evals || [])[index] || { report: r };
    if (slot === 'B') { S.pinB = index; S.compareEval = evalPayload; renderAll(); return; }
    S.pinA = index; S.activeEval = evalPayload; syncFromRegime(index); renderAll();
    if (S.mode === 'live') requestAdvise();
  };
  window.wbClearB = () => { S.pinB = -1; S.compareEval = null; renderAll(); };
  window.wbSwapAB = () => { if (S.pinB < 0) return; [S.pinA, S.pinB] = [S.pinB, S.pinA]; [S.activeEval, S.compareEval] = [S.compareEval, S.activeEval]; renderAll(); };
  window.wbResetSliders = () => { window.wbPinRegime(0, 'A'); window.wbClearB(); };
  window.wbSetView = (m) => {
    el('view-workbench').classList.toggle('view-hidden', m !== 'workbench'); el('view-audit').classList.toggle('view-hidden', m !== 'audit');
    el('btn-view-workbench').classList.toggle('active', m === 'workbench'); el('btn-view-audit').classList.toggle('active', m === 'audit');
  };
  window.wbSetStageTab = (t) => {
    el('wb-stage-arena').style.display = t === 'arena' ? 'flex' : 'none'; el('wb-stage-phase').style.display = t === 'phase' ? 'flex' : 'none';
    el('btn-stage-arena').classList.toggle('active', t === 'arena'); el('btn-stage-phase').classList.toggle('active', t === 'phase');
    if (t === 'phase') requestPhaseMap();
  };
  window.wbSetPhaseMapMode = (m) => { S.phaseMapMode = m === 'core_ready' ? 'core_ready' : 'recommended'; renderPhaseMap(); };
  window.wbSelectPhaseCell = (tokens, context) => {
    const phase = (el('ctl-phase') && el('ctl-phase').value) || 'decode';
    const dp = Math.max(1, num('ctl-dp', 1));
    const batchEl = el('ctl-batch'), seqTokEl = el('ctl-seq-tokens'), ctxEl = el('ctl-context');
    if (phase === 'prefill') {
      if (seqTokEl) { seqTokEl.max = String(Math.max(Number(seqTokEl.max || 8192), tokens)); seqTokEl.value = String(tokens); }
      if (batchEl) { batchEl.max = String(Math.max(Number(batchEl.max || 1024), dp)); batchEl.value = String(dp); }
    } else {
      let batchVal = tokens * dp;
      if (phase === 'verify') {
        const specTok = Math.max(1, num('ctl-seq-tokens', 2) - 1);
        if (seqTokEl && num('ctl-seq-tokens', 1) < 2) seqTokEl.value = String(specTok + 1);
        batchVal = Math.max(1, Math.ceil(tokens / (specTok + 1))) * dp;
      }
      if (batchEl) { batchEl.max = String(Math.max(Number(batchEl.max || 1024), batchVal)); batchEl.value = String(batchVal); }
    }
    if (ctxEl) { ctxEl.max = String(Math.max(Number(ctxEl.max || 1048576), context)); ctxEl.value = String(context); }
    window.wbSetStageTab('arena');
    requestAdvise();
  };
  window.wbApplyLittlesLaw = () => {
    const tps = num('ctl-out-tps', 0), tpot = num('ctl-mean-tpot', 0);
    if (!(tps > 0 && tpot > 0)) return;
    const bVal = Math.max(1, Math.round(tps * (tpot / 1000.0))), bEl = el('ctl-batch');
    if (bEl) { bEl.max = String(Math.max(1024, bVal)); bEl.value = String(bVal); }
    requestAdvise();
  };
  window.wbCopyBucketSchedule = () => { if (navigator.clipboard) navigator.clipboard.writeText(JSON.stringify(S.bucketSchedule || {}, null, 2)); };
  window.wbUploadCalibrationFile = async (file) => {
    if (!file) return;
    if (S.mode !== 'live') { setBanner('wb-static-banner', `Trace upload requires live server: run "python -m vllm_torchtpu.megakernel.advisor serve --model ${DATA.model}".`); return; }
    setTxt('wb-cal-status', `Uploading ${file.name}...`);
    const bytes = new Uint8Array(await file.arrayBuffer());
    const chunks = [];
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      chunks.push(String.fromCharCode.apply(null, bytes.subarray(offset, offset + 0x8000)));
    }
    const res = await postApi('/api/calibrate', { filename: file.name, content_base64: btoa(chunks.join('')) });
    if (res.ok) { S.calibratedSource = file.name; setTxt('wb-cal-status', `Calibrated: ${file.name}`); await requestAdvise(); await requestPhaseMap(); }
    else setTxt('wb-cal-status', `Error: ${res.data.error || 'calibration failed'}`);
  };
  window.wbOnPickCalibration = (ev) => { const f = ev.target && ev.target.files && ev.target.files[0]; if (f) window.wbUploadCalibrationFile(f); };
  window.wbOnDropCalibration = (ev) => { ev.preventDefault(); const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0]; if (f) window.wbUploadCalibrationFile(f); };
  window.toggleTheme = () => { document.documentElement.toggleAttribute('data-theme-light'); };
  syncFromRegime(0); renderAll();
})();
